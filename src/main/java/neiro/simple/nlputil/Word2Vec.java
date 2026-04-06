package neiro.simple.nlputil;

import neiro.simple.mlp.old.ver2.dto.Example;
import neiro.simple.mlp.old.ver2.dto.Examples;
import neiro.simple.mlp.old.ver2.model.*;
import neiro.simple.mlp.old.ver2.train.IWeightOptimaizer;
import neiro.simple.mlp.old.ver2.train.Train;
import neiro.simple.mlp.old.ver2.train.TrainUtil;

import java.util.*;
import java.util.stream.Collectors;

/*
Слова — дискретные сущности, и компьютеру сложно понять их смысл. Word2Vec решает эту проблему, преобразуя каждое слово в плотный вектор (список чисел), который отражает его значение. Главный принцип обучения гласит:
Слово характеризуется компанией, которую оно водит.
То есть смысл слова определяется контекстом — окружающими его словами. Например, «котик» и «пёсик» часто встречаются рядом со словами «милый», «лапки», «гавкает/мяукает» — значит, их векторы будут похожими.

Как это работает: два основных подхода
Word2Vec обучается на большом текстовом корпусе и использует простую нейронную сеть с одним скрытым слоем. Есть два варианта архитектуры:

CBOW (Continuous Bag of Words)
    Модель предсказывает текущее слово по его окружению.
    Пример: по словам «кот ___ на ковре» предсказать пропущенное «сидит».
    CBOW быстрее и лучше работает с частотными словами.

Skip-gram
    Наоборот: по текущему слову предсказывается его контекст.
    Пример: по слову «сидит» предсказать вероятные слова вокруг — «кот», «на», «ковре».
    Skip-gram медленнее, но даёт лучшие результаты для редких слов.

В результате каждое слово отображается в вектор (обычно 100–300 чисел). Эти векторы случайны в начале, а в процессе обучения подстраиваются так, чтобы семантически близкие слова оказались рядом в векторном пространстве.

Удивительные свойства
Word2Vec не просто запоминает похожие слова — он улавливает аналогии и отношения благодаря линейной структуре пространства. Например:
Король - Мужчина + Женщина ≈ Королева
Париж - Франция + Италия ≈ Рим
Повар - готовит + строит ≈ Строитель

Векторные арифметические операции отражают скрытые смысловые связи.

Почему это круто?
    Компактность — вместо разреженных векторов размером в десятки тысяч получаем плотные векторы небольшой размерности.
    Смысловая близость — можно искать синонимы, кластеризовать слова по темам.
    Входной слой для других моделей — векторы слов часто подают в RNN, LSTM или трансформеры.

Ограничения
    Одно значение на слово — не различает омонимы (например, «ключ» как дверной и как родник).
    Игнорирует порядок слов внутри контекстного окна.
    Плохо работает с редкими словами — для них не хватает примеров.
    Не понимает синтаксис и грамматику на более глубоком уровне.

Современный контекст
Сегодня Word2Vec часто уступает место более мощным моделям вроде BERT, ELMo или GPT, которые дают контекстно-зависимые векторы (одно слово может иметь разные векторы в разных предложениях). Однако Word2Vec остаётся отличным выбором для небольших проектов, быстрого прототипирования, задач поиска и классификации, где важны скорость и интерпретируемость.
Если захотите углубиться — стоит посмотреть на аналоги: GloVe (от Stanford) и FastText (от Facebook, учитывает морфологию слов).
*/
public class Word2Vec {

    public static void main(String[] args) {
        String text = """
                Прошу простить меня за обращенье в прозе! Ромашка скромная сказала пышной Розе. Но вижу я: вкруг вашего стебля Живет и множится растительная тля, Мне кажется, что в ней для вас угроза!
                Где вам судить о нас! вспылила Роза. Ромашкам полевым в дела садовых роз Не следует совать свой нос!
                Довольная собой и всех презрев при этом, Красавица погибла тем же летом, Не потому, что рано отцвела, А потому, что дружеским советом Цветка незнатного она пренебрегла… Кто на других глядит высокомерно, Тот этой басни не поймет, наверно
                """;

        Map<String, double[]> vectors = Word2Vec.train(
                text,
                10,    // размер вектора
                2,     // окно
                50,    // эпохи
                0.025  // learning rate
        );

        System.out.println(Arrays.toString(vectors.get("ромашка")));

        // Расстояние между king и queen
        //printCosineDistance(vectors, "king", "queen");

// Ближайшие слова к king
        List<Map.Entry<String, Double>> nearest = findNearestWords(vectors, "ромашка", 5);
        System.out.println("Nearest to:");
        for (var e : nearest) {
            System.out.printf("%s : %.4f\n", e.getKey(), e.getValue());
        }
    }

    /**
     * Обучает word2vec (Skip-gram с полным softmax) на основе вашей MLP-реализации.
     *
     * @param text        обучающий текст
     * @param vectorSize  размерность вектора слова (размер скрытого слоя)
     * @param windowSize  размер контекстного окна (слева и справа)
     * @param epochs      количество эпох
     * @param learningRate скорость обучения (используется IWeightOptimaizer.classic)
     * @return словарь слово -> вектор (double[])
     */
    public static Map<String, double[]> train(String text, int vectorSize, int windowSize, int epochs, double learningRate) {
        // 1. Токенизация и построение словаря
        String[] tokens = text.toLowerCase().split("[^\\p{L}]+");
        Map<String, Integer> wordToIdx = new HashMap<>();
        List<String> idxToWord = new ArrayList<>();
        for (String w : tokens) {
            if (!wordToIdx.containsKey(w)) {
                wordToIdx.put(w, idxToWord.size());
                idxToWord.add(w);
            }
        }
        int vocabSize = idxToWord.size();
        System.out.println("Словарь: " + vocabSize + " слов");

        // 2. Формирование обучающих пар (center, context) в виде Example[]
        //для ускорения можно использовать Negative Sampling
        List<int[]> pairList = new ArrayList<>();
        for (int i = 0; i < tokens.length; i++) {
            int start = Math.max(0, i - windowSize);
            int end = Math.min(tokens.length - 1, i + windowSize);
            for (int j = start; j <= end; j++) {
                if (j == i) continue;
                int centerIdx = wordToIdx.get(tokens[i]);
                int contextIdx = wordToIdx.get(tokens[j]);
                pairList.add(new int[]{centerIdx, contextIdx});
            }
        }
        Example[] trainData = new Example[pairList.size()];
        for (int i = 0; i < pairList.size(); i++) {
            int[] pair = pairList.get(i);
            double[] input = new double[vocabSize];
            double[] target = new double[vocabSize];
            input[pair[0]] = 1.0;
            target[pair[1]] = 1.0;
            trainData[i] = new Example(input, target);
        }
        System.out.println("Сгенерировано " + trainData.length + " обучающих пар");

        // 3. Создание нейросети: входной слой (one-hot), скрытый слой (identity), выходной (softmax+cross-entropy)
        INetMLP net = new Net(List.of(
                //input
                (layer) -> new Layer.input(vocabSize),
                //medium
                (layer) -> new Layer.medium(
                        vectorSize,
                        Fun.INIT_XAVIER(layer.neirons.size(), vectorSize),
                        layer,
                        Fun.IDENTITY,
                        Fun.IDENTITY_DERIVATIVE
                ),
                //output
                (layer) -> new Layer.output.softmaxAndCrossEntity(vocabSize, layer)
        ));

        // 4. Подготовка Examples (только обучение, без теста)
        Examples examples = new Examples(trainData, new Example[0]);

        // 5. Запуск обучения через Train.train
        Train.train(
                net,
                epochs,
                examples,
                TrainUtil.estimationMax,   // не используется, но требуется
                0.05,                      // не используется, но требуется
                new IWeightOptimaizer.classic(learningRate),
                -1,                        // batchSize = -1 (без пакетов)
                10000                      // печать каждые 10000 шагов
        );

        // 6. Извлечение векторов слов из весов входного слоя → скрытый слой
        Net concreteNet = (Net) net;
        Layer.input inputLayer = (Layer.input) concreteNet.layers.get(0);
        Layer.medium hiddenLayer = (Layer.medium) concreteNet.layers.get(1);
        Map<String, double[]> wordVectors = new HashMap<>();

        for (int i = 0; i < vocabSize; i++) {
            Neiron inputNeuron = inputLayer.neirons.get(i);
            double[] vec = new double[vectorSize];
            for (Link link : inputNeuron.oLinks) {
                int hiddenIdx = hiddenLayer.neirons.indexOf(link.oNeiron);
                if (hiddenIdx >= 0 && hiddenIdx < vectorSize) {
                    vec[hiddenIdx] = link.weight.item;
                }
            }
            wordVectors.put(idxToWord.get(i), vec);
        }

        return wordVectors;
    }


    /**
     * Косинусная близость между двумя векторами.
     * Возвращает значение от -1 до 1, где 1 — максимальное сходство.
     */
    public static double cosineSimilarity(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Векторы разной размерности");
        }
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0.0 || normB == 0.0) return 0.0;
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Косинусное расстояние = 1 - косинусная близость.
     * Расстояние 0 означает идентичность, 2 — противоположность.
     */
    public static double cosineDistance(double[] a, double[] b) {
        return 1.0 - cosineSimilarity(a, b);
    }

    /**
     * Поиск topK ближайших слов к заданному слову по косинусной близости.
     * Возвращает список пар (слово, близость), отсортированный по убыванию близости.
     * Само слово исключается из результата.
     */
    public static List<Map.Entry<String, Double>> findNearestWords(
            Map<String, double[]> wordVectors,
            String word,
            int topK) {
        double[] targetVec = wordVectors.get(word);
        if (targetVec == null) {
            throw new IllegalArgumentException("Слово '" + word + "' не найдено в словаре");
        }

        List<Map.Entry<String, Double>> similarities = new ArrayList<>();
        for (Map.Entry<String, double[]> entry : wordVectors.entrySet()) {
            if (entry.getKey().equals(word)) continue;
            double sim = cosineSimilarity(targetVec, entry.getValue());
            similarities.add(new AbstractMap.SimpleEntry<>(entry.getKey(), sim));
        }

        similarities.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        return similarities.stream().limit(topK).collect(Collectors.toList());
    }

    /**
     * Пример использования: для заданных слов выводит косинусное расстояние.
     */
    public static void printCosineDistance(Map<String, double[]> vectors, String word1, String word2) {
        double[] v1 = vectors.get(word1);
        double[] v2 = vectors.get(word2);
        if (v1 == null || v2 == null) {
            System.out.println("Одно из слов не найдено");
            return;
        }
        double sim = cosineSimilarity(v1, v2);
        double dist = 1.0 - sim;
        System.out.printf("Cosine proximity between '%s' и '%s': %.4f\n", word1, word2, sim);
        System.out.printf("Cosine distance: %.4f\n", dist);
    }
}

