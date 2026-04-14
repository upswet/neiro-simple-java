package neiro.simple.nlputil.word2vec;


import lombok.extern.slf4j.Slf4j;
import neiro.simple.mlp.Net;
import neiro.simple.nlputil.TextProcessor;

import java.util.*;

import static neiro.simple.nlputil.word2vec.Word2VecUtils.*;


/**Word2Vec с обучением сразу на всех примерах
 * Эмбединги создаются с использованием самописной полносвязанной сети или отдельной для negative sampling*/
@Slf4j
public class Word2Vec {

    public static void main(String[] args) {
        //преобразовать текст в корпус слов
        String[] words = TextProcessor.processFile("d:\\Work\\Project\\0files\\0corpus\\Набоков-Лолита.txt", 5, "[^а-яА-ЯёЁ\\s]"); //Оставляет только русские буквы (А-Я, а-я, Ё, ё) и пробелы
        //String[] words = TextProcessor.processFile("d:\\Work\\Project\\0files\\0corpus\\толстой.txt", 5, "[^а-яА-ЯёЁ\\s]"); //Оставляет только русские буквы (А-Я, а-я, Ё, ё) и пробелы
        //отфильтровать редкие слова
        words = TextProcessor.removeRare(words, 5);
        //получить для слов эмбединги заданной размерности
        //Map<String, float[]> embedings = train(words, 100, 2, Word2VecType.CBOW, 10);
        //Map<String, float[]> embedings = train(words, 100, 2, Word2VecType.SKIP_GRAMM, 10);
        Map<String, float[]> embedings = train(words, 100, 2, Word2VecType.CBOW_NEGATIVE_SAMPLING, 10);
        //Map<String, float[]> embedings = train(words, 100, 2, Word2VecType.SKIPGRAM_NEGATIVE_SAMPLING, 10);

        //проверить что мы там получили
        //лолита
        findNearestWords(embedings, "лолита", 10);
        findNearestWords(embedings, "гумберт", 10);
        findNearestWords(embedings, "сказал", 10);
        //толстой
        //findNearestWords(embedings, "павловна", 10);
        //findNearestWords(embedings, "князь", 10);
        //findNearestWords(embedings, "княгиня", 10);
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

        // Словарь
        Map<String, Integer> wordToIdx = new HashMap<>();
        List<String> idxToWord = new ArrayList<>();
        for (String w : words) {
            if (!wordToIdx.containsKey(w)) {
                wordToIdx.put(w, idxToWord.size());
                idxToWord.add(w);
            }
        }
        int vocabSize = idxToWord.size();
        log.info("Vocabulary size: {}", vocabSize);

        // Эмбеддинги (будут заполнены позже)
        Map<String, float[]> embeddings = new LinkedHashMap<>();
        for (String word : idxToWord) {
            embeddings.put(word, new float[vectorSize]);
        }

        Random rng = new Random();
        int negativeCount = 5;

        // ---- Обычные алгоритмы (без negative sampling) используем оригинальный Net ----
        if (type == Word2VecType.CBOW || type == Word2VecType.SKIP_GRAMM) {
            List<Net.Example> trainDataList = new ArrayList<>();
            if (type == Word2VecType.CBOW) {
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
            } else { // SKIP_GRAMM
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

            log.info("Train pairs count: {}", trainDataList.size());
            List<Net.Example> testDataList = new ArrayList<>();
            for (int i = 0; i < trainDataList.size() / 10; i++)
                testDataList.add(trainDataList.get((int) (Math.random() * trainDataList.size())));

            Collections.shuffle(trainDataList);
            Net.Examples data = new Net.Examples(trainDataList.toArray(Net.Example[]::new), testDataList.toArray(Net.Example[]::new));

            Net net = new Net(
                    List.of(
                            new Net.LayerInput(vocabSize),
                            new Net.LayerMedium(vectorSize, Net.InitFunTypeEnum.XAVIER, Net.ActivationTypeEnum.IDENTITY),
                            new Net.LayerMedium(vocabSize, Net.InitFunTypeEnum.XAVIER, Net.ActivationTypeEnum.SOFTMAX)
                    ),
                    Net.LossTypeEnum.CROSS_ENTROPY
            );
            net.train(epoch, data, Net.estimationMax, 0.01F,
                    new Net.Optimizator().setType(Net.OptimizatorTypeEnum.SSG).setLearningRate(0.02F).init(),
                    -1, 10000);

            // Извлечение эмбеддингов из весов входного слоя -> скрытый
            for (int idx = 0; idx < vocabSize; idx++) {
                float[] emb = embeddings.get(idxToWord.get(idx));
                for (int j = 0; j < vectorSize; j++) {
                    emb[j] = net.linkWeight[1][j][idx];
                }
            }
            return embeddings;
        }

        // ---- Negative Sampling варианты: используем упрощённую SimpleNetWithNegativeSampling ----
        List<SimpleNetWithNegativeSampling.Example> simpleExamples = new ArrayList<>();

        if (type == Word2VecType.CBOW_NEGATIVE_SAMPLING) {
            // CBOW: вход = среднее контекстных слов (вектор размера vocabSize)
            for (int i = 0; i < words.length; i++) {
                int start = Math.max(0, i - windowSize);
                int end = Math.min(words.length - 1, i + windowSize);
                Set<Integer> contextIndices = new HashSet<>();
                for (int j = start; j <= end; j++) {
                    if (j == i) continue;
                    contextIndices.add(wordToIdx.get(words[j]));
                }
                if (contextIndices.isEmpty()) continue;

                float[] input = new float[vocabSize];
                for (int ctx : contextIndices) input[ctx] += 1.0f;
                for (int k = 0; k < vocabSize; k++) input[k] /= contextIndices.size();

                int centerIdx = wordToIdx.get(words[i]);

                // Положительный класс
                int positive = centerIdx;
                // Отрицательные классы (не контекст и не центральное слово)
                Set<Integer> forbidden = new HashSet<>(contextIndices);
                forbidden.add(centerIdx);
                int[] negatives = new int[negativeCount];
                for (int n = 0; n < negativeCount; n++) {
                    int neg;
                    do {
                        neg = rng.nextInt(vocabSize);
                    } while (forbidden.contains(neg));
                    negatives[n] = neg;
                    forbidden.add(neg); // чтобы не повторялись в одном примере
                }
                simpleExamples.add(new SimpleNetWithNegativeSampling.Example(input, positive, negatives));
            }
        }
        if (type == Word2VecType.SKIPGRAM_NEGATIVE_SAMPLING) {
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
                    input[centerIdx] = 1.0f;  // one-hot центрального слова

                    int positive = ctxIdx;
                    Set<Integer> forbidden = new HashSet<>(contextIndices);
                    forbidden.add(centerIdx);
                    int[] negatives = new int[negativeCount];
                    for (int n = 0; n < negativeCount; n++) {
                        int neg;
                        do {
                            neg = rng.nextInt(vocabSize);
                        } while (forbidden.contains(neg));
                        negatives[n] = neg;
                        forbidden.add(neg);
                    }
                    simpleExamples.add(new SimpleNetWithNegativeSampling.Example(input, positive, negatives));
                }
            }
        }

        log.info("Train examples count (negative sampling): {}", simpleExamples.size());

        // Создаём и обучаем SimpleNet
        SimpleNetWithNegativeSampling simpleNetWithNegativeSampling = new SimpleNetWithNegativeSampling(vocabSize, vectorSize, vocabSize, negativeCount, 0.02f);
        for (int ep = 0; ep < epoch; ep++) {
            Collections.shuffle(simpleExamples);
            for (SimpleNetWithNegativeSampling.Example ex : simpleExamples) {
                simpleNetWithNegativeSampling.train(ex.input, ex.positiveClass, ex.negativeClasses);
            }
            if (ep % 10 == 0) {
                log.info("Epoch {} completed", ep);
            }
        }

        // Извлекаем эмбеддинги из весов w1 (скрытый слой)
        // w1 имеет размер [hiddenSize][inputSize], эмбеддинг слова idx — это столбец w1[0..hiddenSize-1][idx]
        for (int idx = 0; idx < vocabSize; idx++) {
            float[] emb = embeddings.get(idxToWord.get(idx));
            for (int j = 0; j < vectorSize; j++) {
                emb[j] = simpleNetWithNegativeSampling.getWeight1(j, idx);
            }
        }
        return embeddings;
    }



}

