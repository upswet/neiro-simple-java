package neiro.simple.nlputil.word2vec;

import lombok.extern.slf4j.Slf4j;
import neiro.simple.nlputil.TextProcessor;

import java.util.*;
import java.util.stream.Collectors;
import java.util.logging.Logger;

import static neiro.simple.nlputil.word2vec.Word2VecUtils.*;

/**
 * Реализация Word2Vec на чистой Java написанная нейросетью
 * Поддерживает CBOW и Skip-gram с опциональным negative sampling.
 * Использует потоковые итераторы для избежания загрузки всех обучающих примеров в память.
 */
@Slf4j
public class Word2VecDeepSeek {

    public static void main(String[] args) {
        String[] words = TextProcessor.processFile("d:\\Work\\Project\\0files\\0corpus\\толстой.txt", 5, "[^а-яА-ЯёЁ\\s]"); //Оставляет только русские буквы (А-Я, а-я, Ё, ё) и пробелы
        words = TextProcessor.removeRare(words, 10);

        //Map<String, float[]> embeddings = train(words, 100, 2, ModelType.CBOW, 1, 0, 0.02F);
        //Map<String, float[]> embeddings = train(words, 100, 2, ModelType.CBOW, 50, 5, 0.02F);
        //Map<String, float[]> embeddings = train(words, 100, 2, ModelType.SKIP_GRAM, 1, 0, 0.02F);
        Map<String, float[]> embeddings = train(words, 100, 2, ModelType.SKIP_GRAM, 50, 5, 0.02F);

        findNearestWords(embeddings, "павловна", 10);
        findNearestWords(embeddings, "князь", 10);
        findNearestWords(embeddings, "княгиня", 10);
        findNearestWords(embeddings, "мужчина", 10);
        findNearestWords(embeddings, "женщина", 10);

        // Аналогия: король - мужчина + женщина = ?
        List<Word2VecUtils.WordCoeff> expr = Arrays.asList(
                new Word2VecUtils.WordCoeff("князь", 1f),
                new Word2VecUtils.WordCoeff("мужчина", -1f),
                new Word2VecUtils.WordCoeff("молодая", 1f)
        );
        float[] resultVec = computeVector(embeddings, expr);
        List<Map.Entry<String, Double>> nearest = findNearestToVector(embeddings, resultVec, 5, Set.of());
        log.info("Ближайшие к 'князь - мужчина + молодая':");
        nearest.forEach(entry -> log.info("{} : {}", entry.getKey(), entry.getValue()));
    }

    // ------------------------------ Публичное API ------------------------------

    public enum ModelType { CBOW, SKIP_GRAM }

    /**
     * Обучает модель Word2Vec.
     *
     * @param corpus      массив токенов (уже предобработан, редкие слова удалены)
     * @param vectorSize  размерность векторов слов
     * @param windowSize  размер контекстного окна (слева и справа)
     * @param type        CBOW или SKIP_GRAM
     * @param epochs      количество эпох обучения
     * @param negative    количество отрицательных примеров (0 = использовать softmax, >0 = negative sampling)
     * @param learningRate начальная скорость обучения
     * @return отображение "слово -> вектор"
     */
    public static Map<String, float[]> train(String[] corpus, int vectorSize, int windowSize,
                                             ModelType type, int epochs, int negative, float learningRate) {
        // Построение словаря
        Map<String, Integer> wordToIdx = new HashMap<>();
        List<String> idxToWord = new ArrayList<>();
        for (String w : corpus) {
            if (!wordToIdx.containsKey(w)) {
                wordToIdx.put(w, idxToWord.size());
                idxToWord.add(w);
            }
        }
        int vocabSize = idxToWord.size();
        log.info(String.format("Размер словаря: %d", vocabSize));

        // Инициализация векторов слов (входная и выходная матрицы)
        float[][] inputVectors = new float[vocabSize][vectorSize];
        float[][] outputVectors = new float[vocabSize][vectorSize];
        Random rand = new Random(42);
        for (int i = 0; i < vocabSize; i++) {
            for (int j = 0; j < vectorSize; j++) {
                inputVectors[i][j] = (rand.nextFloat() - 0.5f) / vectorSize;
                outputVectors[i][j] = (rand.nextFloat() - 0.5f) / vectorSize;
            }
        }

        // Создание итератора для обучающих примеров
        Iterator<TrainingExample> iterator;
        if (negative == 0) {
            iterator = new SoftmaxExampleIterator(corpus, windowSize, type, wordToIdx);
        } else {
            iterator = new NegativeSamplingExampleIterator(corpus, windowSize, type, wordToIdx, negative, rand);
        }

        int totalExamples = iterator.totalExamples(epochs);
        log.info(String.format("Всего обучающих примеров: %d", totalExamples));

        float lr = learningRate;
        for (int epoch = 0; epoch < epochs; epoch++) {
            if (negative == 0) {
                iterator = new SoftmaxExampleIterator(corpus, windowSize, type, wordToIdx);
            } else {
                iterator = new NegativeSamplingExampleIterator(corpus, windowSize, type, wordToIdx, negative, rand);
            }
            long start = System.currentTimeMillis();
            int processed = 0;
            while (iterator.hasNext()) {
                TrainingExample ex = iterator.next();
                if (type == ModelType.CBOW) {
                    if (negative == 0)
                        trainCbowSoftmax(ex, inputVectors, outputVectors, lr, vocabSize, vectorSize);
                    else
                        trainCbowNegativeSampling(ex, inputVectors, outputVectors, lr, vocabSize, vectorSize, negative);
                } else {
                    if (negative == 0)
                        trainSkipGramSoftmax(ex, inputVectors, outputVectors, lr, vocabSize, vectorSize);
                    else
                        trainSkipGramNegativeSampling(ex, inputVectors, outputVectors, lr, vocabSize, vectorSize, negative);
                }
                processed++;
                if (processed % 100000 == 0) {
                    log.info(String.format("Эпоха %d: обработано %d / %d примеров", epoch+1, processed, totalExamples/epochs));
                }
            }
            long elapsed = System.currentTimeMillis() - start;
            log.info(String.format("Эпоха %d завершена за %d мс, lr=%.5f", epoch+1, elapsed, lr));
            lr *= 0.95;
        }

        Map<String, float[]> result = new LinkedHashMap<>();
        for (int i = 0; i < vocabSize; i++) {
            result.put(idxToWord.get(i), inputVectors[i]);
        }
        return result;
    }

    // ------------------------------ Процедуры обучения ------------------------------

    private static void trainCbowSoftmax(TrainingExample ex, float[][] inputVectors, float[][] outputVectors,
                                         float lr, int vocabSize, int vectorSize) {
        int[] context = ex.contextIndices;
        int target = ex.targetIdx;
        if (context.length == 0) return;

        float[] hidden = new float[vectorSize];
        for (int ctx : context) {
            for (int i = 0; i < vectorSize; i++) hidden[i] += inputVectors[ctx][i];
        }
        for (int i = 0; i < vectorSize; i++) hidden[i] /= context.length;

        float[] scores = new float[vocabSize];
        for (int w = 0; w < vocabSize; w++) {
            float dot = 0;
            for (int i = 0; i < vectorSize; i++) dot += hidden[i] * outputVectors[w][i];
            scores[w] = dot;
        }
        float max = Float.NEGATIVE_INFINITY;
        for (float s : scores) if (s > max) max = s;
        float sum = 0;
        for (int w = 0; w < vocabSize; w++) {
            scores[w] = (float) Math.exp(scores[w] - max);
            sum += scores[w];
        }
        for (int w = 0; w < vocabSize; w++) scores[w] /= sum;

        for (int w = 0; w < vocabSize; w++) {
            float error = scores[w] - (w == target ? 1.0f : 0.0f);
            if (error == 0) continue;
            for (int i = 0; i < vectorSize; i++) {
                outputVectors[w][i] -= lr * error * hidden[i];
            }
            for (int ctx : context) {
                for (int i = 0; i < vectorSize; i++) {
                    inputVectors[ctx][i] -= lr * error * outputVectors[w][i] / context.length;
                }
            }
        }
    }

    private static void trainCbowNegativeSampling(TrainingExample ex, float[][] inputVectors, float[][] outputVectors,
                                                  float lr, int vocabSize, int vectorSize, int negative) {
        int[] context = ex.contextIndices;
        int target = ex.targetIdx;
        int[] negativeWords = ex.negativeIndices;
        if (context.length == 0) return;

        float[] hidden = new float[vectorSize];
        for (int ctx : context) {
            for (int i = 0; i < vectorSize; i++) hidden[i] += inputVectors[ctx][i];
        }
        for (int i = 0; i < vectorSize; i++) hidden[i] /= context.length;

        float dotPos = 0;
        for (int i = 0; i < vectorSize; i++) dotPos += hidden[i] * outputVectors[target][i];
        float sigPos = sigmoid(dotPos);
        float errorPos = sigPos - 1.0f;
        for (int i = 0; i < vectorSize; i++) {
            outputVectors[target][i] -= lr * errorPos * hidden[i];
        }
        for (int ctx : context) {
            for (int i = 0; i < vectorSize; i++) {
                inputVectors[ctx][i] -= lr * errorPos * outputVectors[target][i] / context.length;
            }
        }

        for (int neg : negativeWords) {
            float dotNeg = 0;
            for (int i = 0; i < vectorSize; i++) dotNeg += hidden[i] * outputVectors[neg][i];
            float sigNeg = sigmoid(dotNeg);
            float errorNeg = sigNeg - 0.0f;
            for (int i = 0; i < vectorSize; i++) {
                outputVectors[neg][i] -= lr * errorNeg * hidden[i];
            }
            for (int ctx : context) {
                for (int i = 0; i < vectorSize; i++) {
                    inputVectors[ctx][i] -= lr * errorNeg * outputVectors[neg][i] / context.length;
                }
            }
        }
    }

    private static void trainSkipGramSoftmax(TrainingExample ex, float[][] inputVectors, float[][] outputVectors,
                                             float lr, int vocabSize, int vectorSize) {
        int center = ex.targetIdx;
        int contextWord = ex.contextIndices[0];
        float[] hidden = inputVectors[center];
        float[] scores = new float[vocabSize];
        for (int w = 0; w < vocabSize; w++) {
            float dot = 0;
            for (int i = 0; i < vectorSize; i++) dot += hidden[i] * outputVectors[w][i];
            scores[w] = dot;
        }
        float max = Float.NEGATIVE_INFINITY;
        for (float s : scores) if (s > max) max = s;
        float sum = 0;
        for (int w = 0; w < vocabSize; w++) {
            scores[w] = (float) Math.exp(scores[w] - max);
            sum += scores[w];
        }
        for (int w = 0; w < vocabSize; w++) scores[w] /= sum;

        for (int w = 0; w < vocabSize; w++) {
            float error = scores[w] - (w == contextWord ? 1.0f : 0.0f);
            if (error == 0) continue;
            for (int i = 0; i < vectorSize; i++) {
                outputVectors[w][i] -= lr * error * hidden[i];
            }
            for (int i = 0; i < vectorSize; i++) {
                inputVectors[center][i] -= lr * error * outputVectors[w][i];
            }
        }
    }

    private static void trainSkipGramNegativeSampling(TrainingExample ex, float[][] inputVectors, float[][] outputVectors,
                                                      float lr, int vocabSize, int vectorSize, int negative) {
        int center = ex.targetIdx;
        int contextWord = ex.contextIndices[0];
        int[] negativeWords = ex.negativeIndices;

        float[] hidden = inputVectors[center];

        float dotPos = 0;
        for (int i = 0; i < vectorSize; i++) dotPos += hidden[i] * outputVectors[contextWord][i];
        float sigPos = sigmoid(dotPos);
        float errorPos = sigPos - 1.0f;
        for (int i = 0; i < vectorSize; i++) {
            outputVectors[contextWord][i] -= lr * errorPos * hidden[i];
        }
        for (int i = 0; i < vectorSize; i++) {
            inputVectors[center][i] -= lr * errorPos * outputVectors[contextWord][i];
        }

        for (int neg : negativeWords) {
            float dotNeg = 0;
            for (int i = 0; i < vectorSize; i++) dotNeg += hidden[i] * outputVectors[neg][i];
            float sigNeg = sigmoid(dotNeg);
            float errorNeg = sigNeg - 0.0f;
            for (int i = 0; i < vectorSize; i++) {
                outputVectors[neg][i] -= lr * errorNeg * hidden[i];
            }
            for (int i = 0; i < vectorSize; i++) {
                inputVectors[center][i] -= lr * errorNeg * outputVectors[neg][i];
            }
        }
    }

    private static float sigmoid(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    // ------------------------------ Структуры данных для примеров ------------------------------

    interface Iterator<T> extends java.util.Iterator<T> {
        default int totalExamples(int epochs) { return -1; }
    }

    static class TrainingExample {
        final int[] contextIndices;
        final int targetIdx;
        final int[] negativeIndices;
        TrainingExample(int[] context, int target, int[] negative) {
            this.contextIndices = context;
            this.targetIdx = target;
            this.negativeIndices = negative;
        }
    }

    static class SoftmaxExampleIterator implements Iterator<TrainingExample> {
        private final String[] corpus;
        private final int windowSize;
        private final ModelType type;
        private final Map<String, Integer> wordToIdx;
        private final int[] indices;
        private int pos;

        SoftmaxExampleIterator(String[] corpus, int windowSize, ModelType type, Map<String, Integer> wordToIdx) {
            this.corpus = corpus;
            this.windowSize = windowSize;
            this.type = type;
            this.wordToIdx = wordToIdx;
            this.indices = new int[corpus.length];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            shuffle();
            this.pos = 0;
        }

        private void shuffle() {
            Random r = new Random();
            for (int i = indices.length-1; i > 0; i--) {
                int j = r.nextInt(i+1);
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }

        @Override
        public boolean hasNext() {
            return pos < indices.length;
        }

        @Override
        public TrainingExample next() {
            int wordPos = indices[pos++];
            int centerIdx = wordToIdx.get(corpus[wordPos]);
            if (type == ModelType.CBOW) {
                List<Integer> ctx = new ArrayList<>();
                int start = Math.max(0, wordPos - windowSize);
                int end = Math.min(corpus.length-1, wordPos + windowSize);
                for (int i = start; i <= end; i++) {
                    if (i == wordPos) continue;
                    ctx.add(wordToIdx.get(corpus[i]));
                }
                int[] ctxArr = ctx.stream().mapToInt(Integer::intValue).toArray();
                return new TrainingExample(ctxArr, centerIdx, null);
            } else {
                throw new UnsupportedOperationException("Skip-gram softmax требует особого итератора. Используйте negative sampling.");
            }
        }

        @Override
        public int totalExamples(int epochs) {
            if (type == ModelType.CBOW) return corpus.length * epochs;
            else return corpus.length * (2*windowSize) * epochs;
        }
    }

    static class NegativeSamplingExampleIterator implements Iterator<TrainingExample> {
        private final String[] corpus;
        private final int windowSize;
        private final ModelType type;
        private final Map<String, Integer> wordToIdx;
        private final int negative;
        private final Random rand;
        private final int vocabSize;
        private final int[] indices;
        private int wordIdx;
        private int currentWordPos;
        private List<Integer> contextList;
        private int contextPtr;

        NegativeSamplingExampleIterator(String[] corpus, int windowSize, ModelType type,
                                        Map<String, Integer> wordToIdx, int negative, Random rand) {
            this.corpus = corpus;
            this.windowSize = windowSize;
            this.type = type;
            this.wordToIdx = wordToIdx;
            this.negative = negative;
            this.rand = rand;
            this.vocabSize = wordToIdx.size();
            this.indices = new int[corpus.length];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            shuffle();
            this.wordIdx = -1;
            this.contextPtr = 0;
            moveToNextWord();
        }

        private void shuffle() {
            for (int i = indices.length-1; i > 0; i--) {
                int j = rand.nextInt(i+1);
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }

        private void moveToNextWord() {
            wordIdx++;
            while (wordIdx < indices.length) {
                currentWordPos = indices[wordIdx];
                contextList = getContexts(currentWordPos);
                if (!contextList.isEmpty()) {
                    contextPtr = 0;
                    return;
                }
                wordIdx++;
            }
            contextList = null;
        }

        private List<Integer> getContexts(int pos) {
            List<Integer> ctx = new ArrayList<>();
            int start = Math.max(0, pos - windowSize);
            int end = Math.min(corpus.length-1, pos + windowSize);
            for (int i = start; i <= end; i++) {
                if (i == pos) continue;
                ctx.add(wordToIdx.get(corpus[i]));
            }
            return ctx;
        }

        @Override
        public boolean hasNext() {
            if (contextList != null && contextPtr < contextList.size()) return true;
            if (wordIdx + 1 < indices.length) {
                moveToNextWord();
                return contextList != null && contextPtr < contextList.size();
            }
            return false;
        }

        @Override
        public TrainingExample next() {
            if (!hasNext()) throw new NoSuchElementException();
            int centerIdx = wordToIdx.get(corpus[currentWordPos]);
            if (type == ModelType.CBOW) {
                int[] ctxArr = contextList.stream().mapToInt(Integer::intValue).toArray();
                Set<Integer> negSet = new HashSet<>();
                negSet.add(centerIdx);
                negSet.addAll(contextList);
                int[] negIndices = new int[negative];
                for (int i = 0; i < negative; i++) {
                    int neg;
                    do {
                        neg = rand.nextInt(vocabSize);
                    } while (negSet.contains(neg));
                    negIndices[i] = neg;
                    negSet.add(neg);
                }
                wordIdx++;
                moveToNextWord();
                return new TrainingExample(ctxArr, centerIdx, negIndices);
            } else {
                int ctxWord = contextList.get(contextPtr);
                contextPtr++;
                if (contextPtr >= contextList.size()) {
                    wordIdx++;
                    moveToNextWord();
                }
                Set<Integer> negSet = new HashSet<>();
                negSet.add(centerIdx);
                negSet.add(ctxWord);
                int[] negIndices = new int[negative];
                for (int i = 0; i < negative; i++) {
                    int neg;
                    do {
                        neg = rand.nextInt(vocabSize);
                    } while (negSet.contains(neg));
                    negIndices[i] = neg;
                    negSet.add(neg);
                }
                return new TrainingExample(new int[]{ctxWord}, centerIdx, negIndices);
            }
        }

        @Override
        public int totalExamples(int epochs) {
            if (type == ModelType.CBOW) return corpus.length * epochs;
            else {
                long totalPairs = 0;
                for (int i = 0; i < corpus.length; i++) {
                    int start = Math.max(0, i - windowSize);
                    int end = Math.min(corpus.length-1, i + windowSize);
                    totalPairs += (end - start);
                }
                return (int) (totalPairs * epochs);
            }
        }
    }

}