package neiro.simple.nlputil.word2vec;

import lombok.extern.slf4j.Slf4j;
import neiro.simple.mlp.Net;
import neiro.simple.mlp.NetUtils;
import neiro.simple.nlputil.TextProcessor;

import java.util.*;

import static neiro.simple.nlputil.word2vec.Word2VecUtils.*;

/**Word2Vec с потоковым обучением чтобы память не занимал*/
@Slf4j
public class Word2VecStream {

    public static void main(String[] args) {
        String[] words = TextProcessor.processFile("d:\\Work\\Project\\0files\\0corpus\\Набоков-Лолита.txt", 5, "[^а-яА-ЯёЁ\\s]"); //Оставляет только русские буквы (А-Я, а-я, Ё, ё) и пробелы
        words = TextProcessor.removeRare(words, 5);

        Map<String, float[]> embeddings = train(words, 100, 50, Word2VecType.CBOW, 1);
        //Map<String, float[]> embeddings = train(words, 50, 2, Word2VecType.SKIP_GRAMM, 1);
        //Map<String, float[]> embeddings = train(words, 50, 2, Word2VecType.CBOW_NEGATIVE_SAMPLING, 1);
        //Map<String, float[]> embeddings = train(words, 50, 2, Word2VecType.SKIPGRAM_NEGATIVE_SAMPLING, 1);

        findNearestWords(embeddings, "лолита", 10);
        findNearestWords(embeddings, "гумберт", 10);

        // Аналогия: король - мужчина + женщина = ?
        List<WordCoeff> expr = Arrays.asList(
                new WordCoeff("лолита", 1f),
                new WordCoeff("гумберт", -1f)
        );
        float[] resultVec = computeVector(embeddings, expr);
        List<Map.Entry<String, Double>> nearest = findNearestToVector(embeddings, resultVec, 5, Set.of());
        log.info("Ближайшие к 'лолита - гумберт':");
        nearest.forEach(entry -> log.info("{} : {}", entry.getKey(), entry.getValue()));
    }

    public enum Word2VecType {
        CBOW,
        SKIP_GRAMM,
        CBOW_NEGATIVE_SAMPLING,
        SKIPGRAM_NEGATIVE_SAMPLING
    }

    public static Map<String, float[]> train(String[] words, int vectorSize, int windowSize, Word2VecType type, int epochs) {
        log.info("All words count: {}", words.length);

        // 1. Построение словаря
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

        // 2. Инициализация эмбеддингов (случайные)
        Map<String, float[]> embeddings = new LinkedHashMap<>();
        for (String word : idxToWord) {
            float[] vec = new float[vectorSize];
            for (int i = 0; i < vectorSize; i++) vec[i] = NetUtils.initNormal(0.01f);
            embeddings.put(word, vec);
        }

        // 3. Создание нейронной сети
        boolean useNegativeSampling = (type == Word2VecType.CBOW_NEGATIVE_SAMPLING ||
                type == Word2VecType.SKIPGRAM_NEGATIVE_SAMPLING);
        Net net = new Net(
                List.of(
                        new Net.LayerInput(vocabSize),
                        new Net.LayerMedium(vectorSize, Net.InitFunTypeEnum.XAVIER, Net.ActivationTypeEnum.IDENTITY),
                        new Net.LayerMedium(vocabSize, Net.InitFunTypeEnum.XAVIER,
                                useNegativeSampling ? Net.ActivationTypeEnum.SIGMOID : Net.ActivationTypeEnum.SOFTMAX)
                ),
                Net.LossTypeEnum.CROSS_ENTROPY
        );
        Net.Optimizator optimizator = new Net.Optimizator()
                .setType(Net.OptimizatorTypeEnum.SSG)
                .setLearningRate(0.01F)
                .init();

        // 4. Создание итератора примеров
        int negativeCount = 3;
        Random rng = new Random(42);
        ExampleIterator iterator = new ExampleIterator(words, windowSize, type, wordToIdx, negativeCount, epochs, rng);

        // 5. Потоковое обучение
        int totalExamples = iterator.totalExamples();
        log.info("Total training examples (approx): {}", totalExamples);
        net.trainOnStream(iterator, optimizator, totalExamples, -1, 10000);

        // 6. Извлечение эмбеддингов из весов входного слоя -> скрытый слой
        for (int idx = 0; idx < vocabSize; idx++) {
            String word = idxToWord.get(idx);
            float[] emb = embeddings.get(word);
            for (int j = 0; j < vectorSize; j++) {
                emb[j] = net.linkWeight[1][j][idx];
            }
        }
        return embeddings;
    }

    // ======================== ИТЕРАТОР ПРИМЕРОВ ========================

    /**
     * Итератор, генерирующий обучающие примеры "на лету".
     * Поддерживает все четыре режима Word2Vec и автоматически перебирает эпохи.
     */
    public static class ExampleIterator implements Iterator<Net.Example> {
        private final String[] words;
        private final int windowSize;
        private final Word2VecType type;
        private final Map<String, Integer> wordToIdx;
        private final int vocabSize;
        private final int negativeCount;
        private final Random rng;
        private final int epochs;

        // Управление перебором слов и контекстов
        private int[] indices;          // перемешанные индексы слов в корпусе
        private int wordPos;            // текущая позиция в indices
        private int currentEpoch;
        private List<Integer> currentContexts; // контекстные индексы для текущего слова
        private int ctxIdx;              // индекс внутри currentContexts
        private boolean needNewWord;     // флаг, что нужно перейти к следующему слову

        public ExampleIterator(String[] words, int windowSize, Word2VecType type,
                               Map<String, Integer> wordToIdx, int negativeCount,
                               int epochs, Random rng) {
            this.words = words;
            this.windowSize = windowSize;
            this.type = type;
            this.wordToIdx = wordToIdx;
            this.vocabSize = wordToIdx.size();
            this.negativeCount = negativeCount;
            this.rng = rng;
            this.epochs = epochs;
            this.currentEpoch = 0;

            // Инициализация порядка слов
            this.indices = new int[words.length];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            shuffleIndices();

            this.wordPos = -1;
            this.currentContexts = new ArrayList<>();
            this.ctxIdx = 0;
            this.needNewWord = true;
            moveToNextWord(); // подготовить первое слово
        }

        private void shuffleIndices() {
            for (int i = indices.length - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }

        /**
         * Переход к следующему слову, для которого есть хотя бы один контекст.
         * Если эпоха закончилась, переключается на следующую.
         */
        private void moveToNextWord() {
            while (true) {
                wordPos++;
                if (wordPos >= indices.length) {
                    // Конец текущей эпохи
                    if (currentEpoch + 1 < epochs) {
                        currentEpoch++;
                        shuffleIndices();      // перемешиваем для новой эпохи
                        wordPos = 0;
                    } else {
                        // Все эпохи пройдены
                        currentContexts.clear();
                        needNewWord = false;
                        return;
                    }
                }
                int realPos = indices[wordPos];
                currentContexts = getContextsForWord(realPos);
                if (!currentContexts.isEmpty()) {
                    ctxIdx = 0;
                    needNewWord = false;
                    return;
                }
                // Если контекст пуст (например, windowSize=0), пробуем следующее слово
            }
        }

        /**
         * Возвращает индексы контекстных слов для данной позиции в корпусе.
         */
        private List<Integer> getContextsForWord(int pos) {
            List<Integer> ctx = new ArrayList<>();
            int start = Math.max(0, pos - windowSize);
            int end = Math.min(words.length - 1, pos + windowSize);
            for (int j = start; j <= end; j++) {
                if (j == pos) continue;
                ctx.add(wordToIdx.get(words[j]));
            }
            return ctx;
        }

        @Override
        public boolean hasNext() {
            if (!needNewWord) return true;
            if (wordPos < indices.length - 1 || currentEpoch + 1 < epochs) {
                moveToNextWord();
                return !needNewWord;
            }
            return false;
        }

        @Override
        public Net.Example next() {
            if (!hasNext()) throw new NoSuchElementException();
            int realPos = indices[wordPos];
            int centerIdx = wordToIdx.get(words[realPos]);

            Net.Example ex;
            switch (type) {
                case CBOW:
                    ex = generateCbowExample(realPos, centerIdx);
                    needNewWord = true;   // для CBOW одно слово -> один пример
                    break;
                case CBOW_NEGATIVE_SAMPLING:
                    ex = generateCbowNegativeExample(realPos, centerIdx);
                    needNewWord = true;
                    break;
                case SKIP_GRAMM:
                    int ctxIdxWord = currentContexts.get(ctxIdx);
                    ex = generateSkipGramExample(centerIdx, ctxIdxWord, false);
                    ctxIdx++;
                    if (ctxIdx >= currentContexts.size()) needNewWord = true;
                    break;
                case SKIPGRAM_NEGATIVE_SAMPLING:
                    int ctxIdxNeg = currentContexts.get(ctxIdx);
                    ex = generateSkipGramNegativeExample(realPos, centerIdx, ctxIdxNeg);
                    ctxIdx++;
                    if (ctxIdx >= currentContexts.size()) needNewWord = true;
                    break;
                default:
                    throw new IllegalStateException("Unsupported type");
            }
            return ex;
        }

        // ---------- Генерация примеров для каждого типа ----------

        private Net.Example generateCbowExample(int pos, int centerIdx) {
            int start = Math.max(0, pos - windowSize);
            int end = Math.min(words.length - 1, pos + windowSize);
            float[] input = new float[vocabSize];
            int ctxCount = 0;
            for (int j = start; j <= end; j++) {
                if (j == pos) continue;
                input[wordToIdx.get(words[j])] += 1.0f;
                ctxCount++;
            }
            if (ctxCount > 0) {
                for (int k = 0; k < vocabSize; k++) input[k] /= ctxCount;
            }
            float[] target = new float[vocabSize];
            target[centerIdx] = 1.0f;
            return new Net.Example(input, target);
        }

        private Net.Example generateCbowNegativeExample(int pos, int centerIdx) {
            int start = Math.max(0, pos - windowSize);
            int end = Math.min(words.length - 1, pos + windowSize);
            Set<Integer> contextIndices = new HashSet<>();
            for (int j = start; j <= end; j++) {
                if (j == pos) continue;
                contextIndices.add(wordToIdx.get(words[j]));
            }
            if (contextIndices.isEmpty()) {
                // fallback – просто вернём пустой пример (но такого не должно быть)
                return new Net.Example(new float[vocabSize], new float[vocabSize]);
            }
            float[] input = new float[vocabSize];
            for (int ctx : contextIndices) input[ctx] += 1.0f;
            for (int k = 0; k < vocabSize; k++) input[k] /= contextIndices.size();

            float[] target = new float[vocabSize];
            Arrays.fill(target, Net.NEGATIVE_SAMPLING_VALUE);
            target[centerIdx] = 1.0f;
            for (int n = 0; n < negativeCount; n++) {
                int negIdx;
                do {
                    negIdx = rng.nextInt(vocabSize);
                } while (negIdx == centerIdx || contextIndices.contains(negIdx));
                target[negIdx] = 0f;
            }
            return new Net.Example(input, target);
        }

        private Net.Example generateSkipGramExample(int centerIdx, int ctxIdx, boolean isNegative) {
            float[] input = new float[vocabSize];
            input[centerIdx] = 1.0f;
            float[] target = new float[vocabSize];
            if (!isNegative) {
                target[ctxIdx] = 1.0f;
            } else {
                Arrays.fill(target, Net.NEGATIVE_SAMPLING_VALUE);
                target[ctxIdx] = 1.0f;
                // отрицательные примеры будут добавлены отдельно в другом методе
            }
            return new Net.Example(input, target);
        }

        private Net.Example generateSkipGramNegativeExample(int pos, int centerIdx, int ctxIdx) {
            // Сначала позитивный пример
            float[] input = new float[vocabSize];
            input[centerIdx] = 1.0f;
            float[] target = new float[vocabSize];
            Arrays.fill(target, Net.NEGATIVE_SAMPLING_VALUE);
            target[ctxIdx] = 1.0f;   // положительный

            Set<Integer> allContexts = new HashSet<>(currentContexts);
            for (int n = 0; n < negativeCount; n++) {
                int negIdx;
                do {
                    negIdx = rng.nextInt(vocabSize);
                } while (negIdx == centerIdx || allContexts.contains(negIdx));
                target[negIdx] = 0f;
            }
            return new Net.Example(input, target);
        }

        /**
         * Возвращает приблизительное общее количество примеров за все эпохи.
         * Нужно только для логов, можно вернуть 0, если не хотите вычислять.
         */
        public int totalExamples() {
            // Для CBOW: количество слов * эпохи
            if (type == Word2VecType.CBOW || type == Word2VecType.CBOW_NEGATIVE_SAMPLING) {
                return words.length * epochs;
            }
            // Для Skip-gram: для каждого слова в среднем (2*windowSize) контекстов, но может быть меньше по краям
            // Приблизительная оценка
            return words.length * (2 * windowSize) * epochs;
        }
    }
}
