package neiro.simple.nlputil.word2vec;


import lombok.extern.slf4j.Slf4j;
import neiro.simple.mlp.Net;
import neiro.simple.mlp.NetUtils;
import neiro.simple.nlputil.TextProcessor;

import java.util.*;

import static neiro.simple.nlputil.word2vec.Word2VecUtils.*;


/**Word2Vec с обучением сразу на всех примерах (для не слишком больших текстов)*/
@Slf4j
public class Word2Vec {

    public static void main(String[] args) {
        //преобразовать текст в корпус слов
        String[] words = TextProcessor.processFile("d:\\Work\\Project\\0files\\0corpus\\Набоков-Лолита.txt", 5, "[^а-яА-ЯёЁ\\s]"); //Оставляет только русские буквы (А-Я, а-я, Ё, ё) и пробелы
        //отфильтровать редкие слова
        words = TextProcessor.removeRare(words, 5);
        //получить для слов эмбединги заданной размерности
        Map<String, float[]> embedings = train(words, 100, 2, Word2VecType.CBOW, 10);
        //Map<String, float[]> embedings = train(words, 100, 2, Word2VecType.SKIP_GRAMM, 10);
        //Map<String, float[]> embedings = train(words, 100, 2, Word2VecType.CBOW_NEGATIVE_SAMPLING, 10);
        //Map<String, float[]> embedings = train(words, 100, 2, Word2VecType.SKIPGRAM_NEGATIVE_SAMPLING, 10);
        //проверить что мы там получили
        findNearestWords(embedings, "лолита", 10);
        findNearestWords(embedings, "гумберт", 10);
        findNearestWords(embedings, "сказал", 10);
    }

    public static enum Word2VecType{
        CBOW,
        SKIP_GRAMM,
        CBOW_NEGATIVE_SAMPLING,
        SKIPGRAM_NEGATIVE_SAMPLING
    }

    /** Word2Vec
     *
     * @param words       корпус слов на которых обучаем алгоритм
     * @param vectorSize  размерность вектора слова (размер скрытого слоя)
     * @param windowSize  размер контекстного окна (слева и справа)
     * @param type        тип алгоритма
     * @param epoch       кол-во эпох обучения
     * @return словарь слово -> вектор
     */
    public static Map<String, float[]> train(String[] words, int vectorSize, int windowSize, Word2VecType type, int epoch) {
        log.info("all word count: {}", words.length);

        // Строим словарь уникальных слов
        Map<String, Integer> wordToIdx = new HashMap<>(); //слов в индекс
        List<String> idxToWord = new ArrayList<>();       //индекс в слово
        for (String w : words) {
            if (!wordToIdx.containsKey(w)) {
                wordToIdx.put(w, idxToWord.size());
                idxToWord.add(w);
            }
        }
        int vocabSize = idxToWord.size();
        log.info("Vocabulary size: {}", vocabSize);

        // Инициализируем эмбеддинги (случайные)
        Map<String, float[]> embeddings = new LinkedHashMap<>();
        for (String word : idxToWord) {
            float[] vec = new float[vectorSize];
            for (int i = 0; i < vectorSize; i++) vec[i] = NetUtils.initNormal(0.01f);
            embeddings.put(word, vec);
        }

        // Формируем обучающие примеры в зависимости от типа
        List<Net.Example> trainDataList = new ArrayList<>();
        Random rng = new Random();
        int negativeCount = 5;

        switch (type){
            case CBOW -> {
                // ---- Обычный CBOW (softmax) с усреднением ----
                // на вход подаётся сумма (или среднее) one‑hot векторов контекстных слов
                // на выходе ожидаем вектор центрального слова
                for (int i = 0; i < words.length; i++) {
                    int start = Math.max(0, i - windowSize);
                    int end = Math.min(words.length - 1, i + windowSize);
                    float[] input = new float[vocabSize];
                    int ctxCount = 0;
                    for (int j = start; j <= end; j++) {
                        if (j == i) continue;
                        input[wordToIdx.get(words[j])] += 1.0f;
                        ctxCount++;
                    }
                    if (ctxCount > 0) {
                        for (int k = 0; k < vocabSize; k++) input[k] /= ctxCount;
                    }
                    float[] target = new float[vocabSize];
                    target[wordToIdx.get(words[i])] = 1.0f;
                    trainDataList.add(new Net.Example(input, target));
                }
            }
            case SKIP_GRAMM -> {
                // ---- Обычный Skip-gram (softmax) ----
                //На вход подаётся one‑hot вектор центрального слова
                // на выходе ожидается последовательно one‑hot вектора каждого из контексных слов
                for (int i = 0; i < words.length; i++) {
                    int start = Math.max(0, i - windowSize);
                    int end = Math.min(words.length - 1, i + windowSize);
                    for (int j = start; j <= end; j++) {
                        if (j == i) continue;
                        float[] input = new float[vocabSize];
                        input[wordToIdx.get(words[i])] = 1.0f;
                        float[] target = new float[vocabSize];
                        target[wordToIdx.get(words[j])] = 1.0f;
                        trainDataList.add(new Net.Example(input, target));
                    }
                }
            }
            case CBOW_NEGATIVE_SAMPLING -> {
                // ---- CBOW с negative sampling ----
                // Вход: среднее контекстных слов (вектор признаков размера vocabSize)
                // Выход: сигмоид для целевого слова и negativeCount отрицательных слов
                for (int i = 0; i < words.length; i++) {
                    int start = Math.max(0, i - windowSize);
                    int end = Math.min(words.length - 1, i + windowSize);
                    // Собираем контекстные индексы (окружение)
                    Set<Integer> contextIndices = new HashSet<>();
                    for (int j = start; j <= end; j++) {
                        if (j == i) continue;
                        contextIndices.add(wordToIdx.get(words[j]));
                    }
                    if (contextIndices.isEmpty()) continue;
                    // Вычисляем усреднённый входной вектор (суммируем one-hot контекста и делим чтобы входной вектор остался one-hot)
                    float[] input = new float[vocabSize];
                    for (int ctx : contextIndices) input[ctx] += 1.0f;
                    for (int k = 0; k < vocabSize; k++) input[k] /= contextIndices.size();

                    int centerIdx = wordToIdx.get(words[i]);
                    // Positive пример: target = centerIdx с меткой 1
                    float[] target = new float[vocabSize];
                    Arrays.fill(target, Net.NEGATIVE_SAMPLING_VALUE);
                    target[centerIdx] = 1.0f;
                    // Добавляем отрицательные слова (не из контекста и не само центральное слово)
                    for (int n = 0; n < negativeCount; n++) {
                        int negIdx;
                        do {
                            negIdx = rng.nextInt(vocabSize);
                        } while (negIdx == centerIdx || contextIndices.contains(negIdx));
                        target[negIdx] = 0f;
                    }
                    trainDataList.add(new Net.Example(input, target));
                }
            }
            case SKIPGRAM_NEGATIVE_SAMPLING -> {
                // ---- Skip-gram с negative sampling ----
                // Вход: one-hot центрального слова
                // Выход: для каждого контекстного слова – положительный пример, плюс отрицательные
                for (int i = 0; i < words.length; i++) {
                    int start = Math.max(0, i - windowSize);
                    int end = Math.min(words.length - 1, i + windowSize);
                    Set<Integer> contextIndices = new HashSet<>();
                    for (int j = start; j <= end; j++) {
                        if (j == i) continue;
                        contextIndices.add(wordToIdx.get(words[j]));
                    }
                    int centerIdx = wordToIdx.get(words[i]);
                    for (int ctxIdx : contextIndices) {
                        float[] input = new float[vocabSize];
                        input[centerIdx] = 1.0f;
                        float[] target = new float[vocabSize];
                        Arrays.fill(target, Net.NEGATIVE_SAMPLING_VALUE);
                        target[ctxIdx] = 1.0f;   // positive
                        for (int n = 0; n < negativeCount; n++) {
                            int negIdx;
                            do {
                                negIdx = rng.nextInt(vocabSize);
                            } while (negIdx == ctxIdx || contextIndices.contains(negIdx));
                            target[negIdx] = 0f;
                        }
                        trainDataList.add(new Net.Example(input, target));
                    }
                }
            }
        }

        log.info("Train pairs count: {}", trainDataList.size());

        //сформулируем тестовый набор в 1/10 от обучающего набора
        List<Net.Example> testDataList = new ArrayList<>();
        for(int i =0; i<trainDataList.size() / 10; i++)
            testDataList.add(trainDataList.get((int) (Math.random() * trainDataList.size())));

        Collections.shuffle(trainDataList);
        Net.Examples data = new Net.Examples(trainDataList.toArray(Net.Example[]::new), testDataList.toArray(Net.Example[]::new));

        // Создаём сеть: для Negative Sampling используем Sigmoid + CrossEntropy
        //для negative sampling выходной слой сети должен использовать сигмоиду (а не softmax), а функция потерь – кросс-энтропию
        boolean useNegativeSampling = (type == Word2VecType.CBOW_NEGATIVE_SAMPLING || type == Word2VecType.SKIPGRAM_NEGATIVE_SAMPLING);
        Net net = new Net(
                List.of(
                        new Net.LayerInput(vocabSize),
                        new Net.LayerMedium(vectorSize, Net.InitFunTypeEnum.XAVIER, Net.ActivationTypeEnum.IDENTITY),
                        new Net.LayerMedium(vocabSize, Net.InitFunTypeEnum.XAVIER,
                                useNegativeSampling ? Net.ActivationTypeEnum.SIGMOID : Net.ActivationTypeEnum.SOFTMAX)
                ),
                Net.LossTypeEnum.CROSS_ENTROPY
        );
        net.train(
                epoch,
                data,
                Net.estimationMax,
                0.01F,
                new Net.Optimizator().setType(Net.OptimizatorTypeEnum.SSG).setLearningRate(0.02F).init(),
                -1,
                10000
        );

        // Извлекаем эмбеддинги из весов входного слоя -> скрытый слой
        // linkWeight[1] описывает связи нейронов входного слоя со скрытым слоем и имеет размер [vectorSize][vocabSize]
        for (int idx = 0; idx < vocabSize; idx++) {
            String word = idxToWord.get(idx);
            float[] emb = embeddings.get(word);
            for (int j = 0; j < vectorSize; j++) {
                emb[j] = net.linkWeight[1][j][idx];
            }
        }
        return embeddings;
    }



}

