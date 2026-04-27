package neiro.simple.nlputil.word2vec;

import lombok.extern.slf4j.Slf4j;
import neiro.simple.mlp.util.FileUtils;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

/**
 * Реализация подсловной (subword) модели FastText с использованием negative sampling.
 * <p>
 * Модель изучает векторные представления слов, основываясь на символьных n-граммах.
 * Каждое слово представляется как сумма векторов составляющих его n-грамм.
 * Обучение выполняется методом skip-gram с negative sampling.
 * </p>
 * <p>
 * Улучшения:
 * <ul>
 *   <li>Subsampling частых слов — снижает влияние высокочастотных токенов,
 *       перераспределяя градиент в пользу редких и более информативных слов.</li>
 *   <li>Экспоненциальное затухание learning rate — даёт плавную сходимость
 *       без резкого падения lr до нуля в конце обучения.</li>
 * </ul>
 * </p>
 */
@Slf4j
public class FastTextSubword implements Serializable {
    // ===================== Гиперпараметры =====================
    private final int dim;                 // размерность векторов
    private final double initialLr;        // начальный learning rate
    private final double gamma;            // коэффициент экспоненциального затухания lr (0..1)
    private final int window;              // окно skip-gram
    private final int minn, maxn;          // границы n-грамм (длина от minn до maxn)
    private final int negSamples;          // количество отрицательных примеров
    private final int tableSize = 10_000_000; // размер таблицы для unigram-сэмплирования

    // Параметр subsampling: порог отбрасывания частых слов (классическое t=1e-5)
    private final double subsamplingThreshold = 1e-5;

    // ===================== Словари и матрицы =====================
    // Индексы для n-грамм и входная матрица
    private Map<String, Integer> ngramIndex = new HashMap<>();
    private double[][] wi;                 // матрица входных векторов n-грамм [ngramCount][dim]

    // Индексы для слов и выходная матрица
    private Map<String, Integer> wordIndex = new HashMap<>();
    private double[][] wo;                 // матрица выходных векторов слов [wordCount][dim]

    /**
     * Таблица для быстрого сэмплирования отрицательных примеров в соответствии
     * с распределением unigram в степени 0.75 (сглаживание частот).
     */
    private int[] unigramTable;

    /**
     * Вероятность сохранить слово при subsampling (от 0 до 1).
     * Чем чаще слово, тем ниже вероятность (но не ниже 0).
     */
    private Map<String, Double> wordKeepProb = new HashMap<>();

    /**
     * Общее количество токенов в корпусе (для вычисления частот).
     */
    private long totalTokens = 0;

    private Random rnd = new Random(42);

    // Сохранение средних косинусных расстояний для критерия ранней остановки (опционально)
    double analogiesAvgPrev;

    // ===================== Конструктор =====================

    /**
     * Конструктор модели FastText.
     *
     * @param dim         размерность векторного пространства
     * @param initialLr   начальная скорость обучения
     * @param gamma       коэффициент экспоненциального затухания lr (обычно 0.9..0.99).
     *                    При gamma=1.0 получится константный lr, при gamma=0.9 — быстрое затухание.
     * @param window      размер контекстного окна (сколько слов слева и справа учитывать)
     * @param minn        минимальная длина символьной n-граммы
     * @param maxn        максимальная длина символьной n-граммы
     * @param negSamples  число отрицательных примеров на один положительный
     */
    public FastTextSubword(int dim, double initialLr, double gamma, int window,
                           int minn, int maxn, int negSamples) {
        this.dim = dim;
        this.initialLr = initialLr;
        this.gamma = gamma;
        this.window = window;
        this.minn = minn;
        this.maxn = maxn;
        this.negSamples = negSamples;
    }

    // ===================== N-граммы =====================

    /**
     * Извлекает из слова все символьные n-граммы заданного диапазона длин.
     * Слово обрамляется угловыми скобками "<" и ">", чтобы отделить начало и конец.
     * Если по заданным minn/maxn не удалось извлечь ни одной n-граммы,
     * возвращается список, содержащий само обрамлённое слово.
     *
     * @param word исходное слово (без специальных символов)
     * @return список строковых n-грамм, например ["<ma", "mar", ..., "ia>"] для "maria"
     */
    private List<String> extractNgrams(String word) {
        String w = "<" + word + ">";
        List<String> ngrams = new ArrayList<>();
        for (int n = minn; n <= maxn; n++) {
            for (int i = 0; i <= w.length() - n; i++) {
                ngrams.add(w.substring(i, i + n));
            }
        }
        // fallback: если слово слишком короткое или minn/maxn некорректны
        if (ngrams.isEmpty()) {
            ngrams.add(w);
        }
        return ngrams;
    }

    // ===================== Subsampling =====================

    /**
     * Вычисляет вероятность сохранения слова на основе его частоты.
     * Используется формула из оригинального word2vec:
     * P_keep(w) = sqrt(t / freq)  при freq > t, иначе 1.0,
     * где t = subsamplingThreshold (по умолчанию 1e-5).
     *
     * @param freq доля слова в общем количестве токенов (count / totalTokens)
     * @return вероятность оставить слово в предложении (от 0 до 1)
     */
    private double keepProb(double freq) {
        if (freq > subsamplingThreshold) {
            return Math.sqrt(subsamplingThreshold / freq);
        }
        return 1.0;
    }

    /**
     * Применяет subsampling к предложению: каждое слово может быть удалено
     * с вероятностью (1 - wordKeepProb). Слова, отсутствующие в словаре,
     * сохраняются всегда (или можно удалять, но они не участвуют в обучении).
     *
     * @param sentence исходный список слов предложения
     * @return отфильтрованный список слов
     */
    private List<String> subsampleSentence(List<String> sentence) {
        List<String> filtered = new ArrayList<>();
        for (String w : sentence) {
            Double keep = wordKeepProb.get(w);
            // если для слова не задана вероятность (редкое), сохраняем всегда
            if (keep == null || rnd.nextDouble() < keep) {
                filtered.add(w);
            }
        }
        return filtered;
    }

    // ===================== Обучение на одном предложении =====================

    /**
     * Обрабатывает одно предложение: для каждого слова (центра) предсказывает
     * окружающие слова (контекст) с помощью skip-грам с negative sampling.
     * Вектор центра получается суммированием векторов его n-грамм.
     *
     * @param sentence  список слов предложения (уже прошедший subsampling)
     * @param currentLr текущая скорость обучения для этого шага
     */
    public void trainSentence(List<String> sentence, double currentLr) {
        // Для каждого слова в предложении как центрального
        for (int center = 0; center < sentence.size(); center++) {
            String centerWord = sentence.get(center);
            List<String> centerNgrams = extractNgrams(centerWord);
            List<Integer> centerNgramIdx = new ArrayList<>();
            for (String ng : centerNgrams) {
                Integer idx = ngramIndex.get(ng);
                if (idx != null) centerNgramIdx.add(idx);
            }
            if (centerNgramIdx.isEmpty()) continue; // нет известных n-грамм, пропускаем

            // Контекстное окно (слова в радиусе window, исключая само центральное)
            int start = Math.max(0, center - window);
            int end = Math.min(sentence.size(), center + window + 1);
            for (int ctx = start; ctx < end; ctx++) {
                if (ctx == center) continue;
                String contextWord = sentence.get(ctx);
                Integer ctxIdx = wordIndex.get(contextWord);
                if (ctxIdx == null) continue; // слово отсутствует в словаре, пропускаем

                // --- Положительный пример (реальное слово контекста) ---
                // Вычисляем скалярное произведение суммы векторов n-грамм и wo[ctxIdx]
                double dotPos = 0.0;
                for (int d = 0; d < dim; d++) {
                    double sumWi = 0.0;
                    for (int idx : centerNgramIdx) {
                        sumWi += wi[idx][d];
                    }
                    dotPos += sumWi * wo[ctxIdx][d];
                }
                double sigPos = sigmoid(dotPos);
                double gPos = (sigPos - 1) * currentLr;  // градиент: модель должна предсказать 1.0

                // Обновление весов для каждой компоненты
                for (int d = 0; d < dim; d++) {
                    // Текущая сумма входа для этой размерности (актуальное состояние wi)
                    double input_d = 0.0;
                    for (int idx : centerNgramIdx) {
                        input_d += wi[idx][d];
                    }
                    // Сохраняем старое значение выходного вектора
                    double oldWO = wo[ctxIdx][d];
                    // Обновляем выходной вектор
                    wo[ctxIdx][d] -= gPos * input_d;
                    // Обновляем входные векторы n-грамм, используя СТАРОЕ wo
                    for (int idx : centerNgramIdx) {
                        wi[idx][d] -= gPos * oldWO;
                    }
                }

                // --- Отрицательные примеры (слова, не встречающиеся в данном контексте) ---
                for (int n = 0; n < negSamples; n++) {
                    int negIdx;
                    do {
                        // Выбираем случайное слово по unigram-распределению (сглаженному)
                        negIdx = unigramTable[rnd.nextInt(tableSize)];
                    } while (negIdx == ctxIdx); // не должен совпадать с положительным примером

                    // Скалярное произведение для отрицательного слова
                    double dotNeg = 0.0;
                    for (int d = 0; d < dim; d++) {
                        double sumWi = 0.0;
                        for (int idx : centerNgramIdx) {
                            sumWi += wi[idx][d];
                        }
                        dotNeg += sumWi * wo[negIdx][d];
                    }
                    double sigNeg = sigmoid(dotNeg);
                    double gNeg = sigNeg * currentLr; // градиент: модель должна предсказать 0.0

                    // Обновление весов
                    for (int d = 0; d < dim; d++) {
                        double input_d = 0.0;
                        for (int idx : centerNgramIdx) {
                            input_d += wi[idx][d];
                        }
                        double oldWO = wo[negIdx][d];
                        wo[negIdx][d] -= gNeg * input_d;
                        for (int idx : centerNgramIdx) {
                            wi[idx][d] -= gNeg * oldWO;
                        }
                    }
                }
            }
        }
    }

    // ===================== Полный цикл обучения =====================

    /**
     * Полный цикл обучения модели с экспоненциальным затуханием learning rate.
     * Перед каждым обучающим шагом к предложению применяется subsampling.
     * После каждой эпохи вычисляется среднее косинусное сходство для заданных аналогий.
     *
     * @param corpus    список предложений (каждое предложение – список слов)
     * @param epochs    количество эпох обучения
     * @param analogies список аналогий для мониторинга качества
     */
    public void train(List<List<String>> corpus, int epochs, List<Analogy> analogies) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Экспоненциальное затухание learning rate: lr = initialLr * gamma^epoch
            double currentLr = initialLr * Math.pow(gamma, epoch);

            // Обучаем на всех предложениях корпуса с subsampling
            for (List<String> sent : corpus) {
                // Убираем часть частых слов, чтобы сбалансировать влияние
                List<String> filtered = subsampleSentence(sent);
                if (!filtered.isEmpty()) {
                    trainSentence(filtered, currentLr);
                }
            }

            // Оценка качества по аналогиям
            double analogiesAvg = analogies.stream()
                    .map(analogy -> cosine(
                            addVectors(
                                    subtractVectors(
                                            getWordVector(analogy.one()),
                                            getWordVector(analogy.two())
                                    ),
                                    getWordVector(analogy.three())
                            ),
                            getWordVector(analogy.result())
                    ))
                    .reduce(0.0, Double::sum) / analogies.size();

            log.info("Эпоха {}/{} analogiesAvg={}, lr = {}", epoch + 1, epochs, analogiesAvg, currentLr);

            // Ранняя остановка может быть включена при необходимости
            // if (epoch > 0 && analogiesAvgPrev > analogiesAvg) { ... }

            analogiesAvgPrev = analogiesAvg;
        }
    }

    // ===================== Вспомогательные функции =====================

    /**
     * Скалярное произведение двух векторов.
     */
    private double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < dim; i++) s += a[i] * b[i];
        return s;
    }

    /**
     * Логистическая функция (сигмоида).
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // ===================== Построение словарей =====================

    /**
     * Построение словарей и начальная инициализация матриц весов.
     * Собирает все слова из корпуса, присваивает им индексы, инициализирует выходную матрицу wo.
     * Затем собирает все n-граммы слов и инициализирует входную матрицу wi.
     * Также строит unigram-таблицу для сэмплирования отрицательных примеров
     * и вычисляет вероятности сохранения слов для subsampling.
     *
     * @param corpus список предложений
     */
    public void buildVocab(List<List<String>> corpus) {
        log.info("Начинаем построение словаря. Всего предложений: {}", corpus.size());

        // ---- Подсчёт частот слов ----
        Map<String, Integer> wordFreq = new HashMap<>();
        for (List<String> sent : corpus) {
            for (String w : sent) {
                wordFreq.put(w, wordFreq.getOrDefault(w, 0) + 1);
            }
        }

        // Общее количество токенов в корпусе (нужно для subsampling и unigram-таблицы)
        totalTokens = wordFreq.values().stream().mapToLong(Integer::longValue).sum();
        log.debug("Всего токенов: {}", totalTokens);

        // ---- Присвоение индексов словам и инициализация wo случайными малыми значениями ----
        int wIdx = 0;
        for (String w : wordFreq.keySet()) {
            wordIndex.put(w, wIdx++);
        }
        wo = new double[wordIndex.size()][dim];
        for (int i = 0; i < wo.length; i++) {
            for (int d = 0; d < dim; d++) {
                wo[i][d] = (rnd.nextDouble() - 0.5) / dim;
            }
        }

        // ---- Сбор всех n-грамм и инициализация wi ----
        for (List<String> sent : corpus) {
            for (String w : sent) {
                for (String ng : extractNgrams(w)) {
                    ngramIndex.putIfAbsent(ng, ngramIndex.size());
                }
            }
        }
        wi = new double[ngramIndex.size()][dim];
        for (int i = 0; i < wi.length; i++) {
            for (int d = 0; d < dim; d++) {
                wi[i][d] = (rnd.nextDouble() - 0.5) / dim;
            }
        }

        // ---- Построение unigram-таблицы для negative sampling ----
        // Используется распределение частот слов, возведённое в степень 0.75,
        // что уменьшает вероятность слишком частых слов и увеличивает для редких.
        double[] pw = new double[wordFreq.size()];
        int idx = 0;
        double sum = 0;
        for (Map.Entry<String, Integer> e : wordFreq.entrySet()) {
            pw[idx] = Math.pow(e.getValue(), 0.75);
            sum += pw[idx];
            idx++;
        }
        unigramTable = new int[tableSize];
        int tablePos = 0;
        double acc = 0;
        for (Map.Entry<String, Integer> e : wordFreq.entrySet()) {
            acc += Math.pow(e.getValue(), 0.75) / sum;
            int wordIdx = wordIndex.get(e.getKey());
            int cnt = (int) (acc * tableSize) - tablePos;
            for (int i = 0; i < cnt; i++) {
                unigramTable[tablePos++] = wordIdx;
            }
        }
        // На случай ошибок округления заполняем оставшиеся ячейки индексом 0 (слово с индексом 0)
        while (tablePos < tableSize) {
            unigramTable[tablePos++] = 0;
        }

        // ---- Вычисление вероятностей для subsampling ----
        wordKeepProb.clear();
        for (Map.Entry<String, Integer> e : wordFreq.entrySet()) {
            double freq = (double) e.getValue() / totalTokens;
            double keep = keepProb(freq);
            // храним только если необходимо отбрасывать (keep < 1.0), но для простоты сохраняем все
            wordKeepProb.put(e.getKey(), keep);
        }

        log.info("Закончили составлять словарь word count={}, ngram count={}",
                wordFreq.size(), ngramIndex.size());
    }

    // ===================== Работа с векторами =====================

    /**
     * Возвращает вектор слова, вычисленный как среднее арифметическое векторов
     * всех его символьных n-грамм. Может использоваться даже для слов, отсутствовавших
     * в обучающей выборке (out-of-vocabulary).
     *
     * @param word слово
     * @return вещественный вектор размерности dim
     */
    public double[] getWordVector(String word) {
        double[] vec = new double[dim];
        List<String> ngrams = extractNgrams(word);
        int cnt = 0;
        for (String ng : ngrams) {
            Integer idx = ngramIndex.get(ng);
            if (idx != null) {
                for (int d = 0; d < dim; d++) {
                    vec[d] += wi[idx][d];
                }
                cnt++;
            }
        }
        if (cnt > 0) {
            for (int d = 0; d < dim; d++) vec[d] /= cnt; // усреднение по всем найденным n-граммам
        }
        return vec;
    }

    /**
     * Поиск N ближайших слов к заданному вектору по косинусному сходству.
     *
     * @param vector опорный вектор
     * @param N      количество ближайших слов для выдачи
     * @return список строк вида "слово(значение_сходства)", отсортированный по убыванию сходства
     */
    public List<String> getNearestWords(double[] vector, int N) {
        List<Map.Entry<String, Double>> scores = new ArrayList<>();
        for (String word : wordIndex.keySet()) {
            double[] wv = getWordVector(word);
            double sim = cosine(vector, wv);
            scores.add(new AbstractMap.SimpleEntry<>(word, sim));
        }
        scores.sort((a, b) -> Double.compare(b.getValue(), a.getValue())); // сортировка по убыванию
        List<String> result = new ArrayList<>();
        for (int i = 0; i < Math.min(N, scores.size()); i++) {
            result.add(scores.get(i).getKey() + "(" + scores.get(i).getValue() + ")");
        }
        return result;
    }

    /**
     * Покомпонентное сложение двух векторов одинаковой длины.
     *
     * @param a первый вектор
     * @param b второй вектор
     * @return новый вектор a + b
     */
    public static double[] addVectors(double[] a, double[] b) {
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] + b[i];
        return res;
    }

    /**
     * Покомпонентное вычитание векторов: a - b.
     *
     * @param a уменьшаемый вектор
     * @param b вычитаемый вектор
     * @return новый вектор a - b
     */
    public static double[] subtractVectors(double[] a, double[] b) {
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] - b[i];
        return res;
    }

    /**
     * Косинусное сходство между двумя векторами.
     *
     * @param a первый вектор
     * @param b второй вектор
     * @return значение косинуса угла между a и b в диапазоне [-1, 1]
     */
    public static double cosine(double[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-10);
    }

    // ===================== Чтение корпуса =====================

    /**
     * Вспомогательный метод чтения корпуса из директории с текстовыми файлами (.txt).
     * Предполагается, что каждый файл содержит по одному предложению на строке.
     * Токенизация: оставляются только буквенные символы, всё приводится к нижнему регистру.
     *
     * @param dirPath путь к директории с .txt файлами
     * @return список предложений (каждое предложение – список токенов-слов)
     * @throws IOException если произошла ошибка ввода-вывода при чтении файлов
     */
    public static List<List<String>> readCorpusFromDirectory(String dirPath) throws IOException {
        List<List<String>> corpus = new ArrayList<>();
        File dir = new File(dirPath);
        File[] files = dir.listFiles((d, name) -> name.endsWith(".txt"));
        if (files == null) return corpus;

        for (File file : files) {
            List<String> lines = Files.readAllLines(file.toPath(), StandardCharsets.UTF_8);
            for (String line : lines) {
                // Простейшая токенизация: удаляем не-буквы, разбиваем по пробелам
                String[] tokens = line.toLowerCase()
                        .replaceAll("[^\\p{L}\\s]", " ")
                        .split("\\s+");
                List<String> sentence = new ArrayList<>();
                for (String t : tokens) {
                    if (!t.isEmpty()) {
                        sentence.add(t);
                    }
                }
                if (!sentence.isEmpty()) {
                    corpus.add(sentence);
                }
            }
        }
        return corpus;
    }

    // ===================== Демонстрация =====================

    /**
     * Демонстрационный пример обучения FastText на мини-корпусе с ручным заданием
     * синонимов и антонимов для контроля качества. Выводит ближайшие слова и
     * проверяет линейные отношения между векторами.
     */
    public static void main(String[] args) throws IOException {
        List<List<String>> externalCorpus = readCorpusFromDirectory("d:\\Work\\Project\\0files\\0corpus\\"); // Чтение корпуса
        
        // Создаём модель с subsampling (параметр gamma = 0.95 для экспоненциального затухания)
        FastTextSubword ft = new FastTextSubword(
                30,    // размерность векторов
                0.02,  // начальный learning rate
                0.95,  // коэффициент экспоненциального затухания lr
                3,     // окно (слова слева и справа)
                3,     // минимальная длина n-граммы
                6,     // максимальная длина n-граммы
                5      // количество отрицательных примеров
        );

        ft.buildVocab(externalCorpus);

        //FileUtils.save("ft1",ft);
        //FastTextSubword ft= (FastTextSubword) FileUtils.load("ft1");

        // Обучение с одной аналогией для мониторинга
        ft.train(externalCorpus, 60, List.of(
                new Analogy("князь", "мужчина", "женщина", "княгиня")
                ,new Analogy("лорд", "мужчина", "женщина", "леди")
                ,new Analogy("гриффиндор", "лев", "змея", "слизерин")
        ));

        // Проверка результатов
        log.info("Ближайшие к 'попа': {}", ft.getNearestWords(ft.getWordVector("попа"), 10));
        log.info("Ближайшие к 'гриффиндор': {}", ft.getNearestWords(ft.getWordVector("гриффиндор"), 10));
        log.info("Ближайшие к 'слизерин': {}", ft.getNearestWords(ft.getWordVector("слизерин"), 10));
        log.info("Ближайшие к 'сказал': {}", ft.getNearestWords(ft.getWordVector("сказал"), 10));
        log.info("Ближайшие к 'сделал': {}", ft.getNearestWords(ft.getWordVector("сделал"), 10));
        log.info("Ближайшие к 'умер': {}", ft.getNearestWords(ft.getWordVector("умер"), 10));
        log.info("Ближайшие к 'мужчина': {}", ft.getNearestWords(ft.getWordVector("мужчина"), 10));
        log.info("Ближайшие к 'женщина': {}", ft.getNearestWords(ft.getWordVector("женщина"), 10));
        log.info("Ближайшие к 'гриффиндор - лев + змея': {}",
                ft.getNearestWords(addVectors(subtractVectors(ft.getWordVector("гриффиндор"), ft.getWordVector("лев")),
                        ft.getWordVector("змея")), 10));
        log.info("Ближайшие к 'лорд - мужчина + женщина': {}",
                ft.getNearestWords(addVectors(subtractVectors(ft.getWordVector("лорд"), ft.getWordVector("мужчина")),
                        ft.getWordVector("женщина")), 10));
        log.info("Ближайшие к 'князь - мужчина + женщина': {}",
                ft.getNearestWords(addVectors(subtractVectors(ft.getWordVector("князь"), ft.getWordVector("мужчина")),
                        ft.getWordVector("женщина")), 10));
        log.info("Ближайшие к 'царь - мужчина + женщина': {}",
                ft.getNearestWords(addVectors(subtractVectors(ft.getWordVector("царь"), ft.getWordVector("мужчина")),
                        ft.getWordVector("женщина")), 10));
        log.info("Расстояние между царём и кошкой': {}",
                cosine(ft.getWordVector("царь"), ft.getWordVector("кошка")));
        log.info("Расстояние между любимый - любимая и мужчина - женщина': {}",
                cosine(
                        subtractVectors(ft.getWordVector("любимый"), ft.getWordVector("любимая")),
                        subtractVectors(ft.getWordVector("мужчина"), ft.getWordVector("женщина"))
                ));

        FileUtils.save("ft1",ft);
    }
}

/**
 * Вспомогательный record для задания аналогии вида one - two + three = result.
 */
record Analogy(String one, String two, String three, String result) {}