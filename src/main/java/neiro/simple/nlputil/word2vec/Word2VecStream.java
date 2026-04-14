package neiro.simple.nlputil.word2vec;

import lombok.extern.slf4j.Slf4j;
import neiro.simple.mlp.Net;
import neiro.simple.mlp.NetUtils;
import neiro.simple.nlputil.TextProcessor;

import java.util.*;

import static neiro.simple.nlputil.word2vec.Word2VecUtils.*;

/**Word2Vec с потоковым обучением, не занимающим память.
 * Для negative sampling используется упрощённая сеть SimpleNetWithNegativeSampling (только SSG, без батчей).*/
@Slf4j
public class Word2VecStream {

    public static void main(String[] args) {
        //String[] words = TextProcessor.processFile("d:\\Work\\Project\\0files\\0corpus\\Набоков-Лолита.txt", 5, "[^а-яА-ЯёЁ\\s]");
        String[] words = TextProcessor.processFile("d:\\Work\\Project\\0files\\0corpus\\толстой.txt", 5, "[^а-яА-ЯёЁ\\s]"); //Оставляет только русские буквы (А-Я, а-я, Ё, ё) и пробелы

        words = TextProcessor.removeRare(words, 5);

        // Пример использования с negative sampling
        Map<String, float[]> embedings = train(words, 100, 2, Word2VecType.CBOW_NEGATIVE_SAMPLING, 1);
        // Map<String, float[]> embedings = train(words, 100, 2, Word2VecType.CBOW, 1);
        // Map<String, float[]> embedings = train(words, 100, 2, Word2VecType.SKIP_GRAMM, 1);
        // Map<String, float[]> embedings = train(words, 100, 2, Word2VecType.SKIPGRAM_NEGATIVE_SAMPLING, 1);

        //проверить что мы там получили
        //лолита
        //findNearestWords(embedings, "лолита", 10);
        //findNearestWords(embedings, "гумберт", 10);
        //findNearestWords(embedings, "сказал", 10);
        //толстой
        findNearestWords(embedings, "павловна", 10);
        findNearestWords(embedings, "князь", 10);
        findNearestWords(embedings, "княгиня", 10);
    }

    public enum Word2VecType {
        CBOW,
        SKIP_GRAMM,
        CBOW_NEGATIVE_SAMPLING,
        SKIPGRAM_NEGATIVE_SAMPLING
    }

    public static Map<String, float[]> train(String[] words, int vectorSize, int windowSize, Word2VecType type, int epochs) {
        log.info("All words count: {}", words.length);

        // Построение словаря
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

        // Контейнер для эмбеддингов (будут заполнены после обучения)
        Map<String, float[]> embeddings = new LinkedHashMap<>();
        for (String word : idxToWord) {
            embeddings.put(word, new float[vectorSize]);
        }

        // ---- Обычные алгоритмы (без negative sampling) используем оригинальный Net ----
        if (type == Word2VecType.CBOW || type == Word2VecType.SKIP_GRAMM) {
            Net net = new Net(
                    List.of(
                            new Net.LayerInput(vocabSize),
                            new Net.LayerMedium(vectorSize, Net.InitFunTypeEnum.XAVIER, Net.ActivationTypeEnum.IDENTITY),
                            new Net.LayerMedium(vocabSize, Net.InitFunTypeEnum.XAVIER, Net.ActivationTypeEnum.SOFTMAX)
                    ),
                    Net.LossTypeEnum.CROSS_ENTROPY
            );
            Net.Optimizator optimizator = new Net.Optimizator()
                    .setType(Net.OptimizatorTypeEnum.SSG)
                    .setLearningRate(0.02F)
                    .init();

            Random rng = new Random(42);
            ExampleIteratorNet iterator = new ExampleIteratorNet(words, windowSize, type, wordToIdx, 0, epochs, rng);
            int totalExamples = iterator.totalExamples();
            log.info("Total training examples (Net): {}", totalExamples);
            net.trainOnStream(iterator, optimizator, totalExamples, -1, 10000);

            // Извлечение эмбеддингов из весов Net (входной слой -> скрытый)
            for (int idx = 0; idx < vocabSize; idx++) {
                float[] emb = embeddings.get(idxToWord.get(idx));
                for (int j = 0; j < vectorSize; j++) {
                    emb[j] = net.linkWeight[1][j][idx];
                }
            }
            return embeddings;
        }

        // ---- Negative Sampling варианты: используем упрощённую SimpleNetWithNegativeSampling ----
        int negativeCount = 5;
        Random rng = new Random(42);

        ExampleIteratorSimple iteratorSimple = new ExampleIteratorSimple(words, windowSize, type, wordToIdx, negativeCount, epochs, rng);
        SimpleNetWithNegativeSampling simpleNet = new SimpleNetWithNegativeSampling(vocabSize, vectorSize, vocabSize, negativeCount, 0.02f);

        // Обучение по эпохам
        for (int ep = 0; ep < epochs; ep++) {
            iteratorSimple.resetEpoch();  // сбрасываем итератор для новой эпохи
            int exampleCount = 0;
            while (iteratorSimple.hasNext()) {
                SimpleNetWithNegativeSampling.Example ex = iteratorSimple.next();
                simpleNet.train(ex.input, ex.positiveClass, ex.negativeClasses);
                exampleCount++;
                if (exampleCount % 10000 == 0) {
                    log.info("Epoch {}, processed {} examples", ep, exampleCount);
                }
            }
            log.info("Epoch {} completed, total examples: {}", ep, exampleCount);
        }

        // Извлечение эмбеддингов из SimpleNet (веса w1)
        for (int idx = 0; idx < vocabSize; idx++) {
            float[] emb = embeddings.get(idxToWord.get(idx));
            for (int j = 0; j < vectorSize; j++) {
                emb[j] = simpleNet.getWeight1(j, idx);
            }
        }
        return embeddings;
    }

    // ======================== ИТЕРАТОР ДЛЯ NET (ОБЫЧНЫЕ РЕЖИМЫ) ========================
    public static class ExampleIteratorNet implements Iterator<Net.Example> {
        private final String[] words;
        private final int windowSize;
        private final Word2VecType type;
        private final Map<String, Integer> wordToIdx;
        private final int vocabSize;
        private final Random rng;
        private final int epochs;
        private int[] indices;
        private int wordPos;
        private int currentEpoch;
        private List<Integer> currentContexts;
        private int ctxIdx;
        private boolean needNewWord;

        public ExampleIteratorNet(String[] words, int windowSize, Word2VecType type,
                                  Map<String, Integer> wordToIdx, int negativeCount,
                                  int epochs, Random rng) {
            this.words = words;
            this.windowSize = windowSize;
            this.type = type;
            this.wordToIdx = wordToIdx;
            this.vocabSize = wordToIdx.size();
            this.rng = rng;
            this.epochs = epochs;
            this.currentEpoch = 0;
            this.indices = new int[words.length];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            shuffleIndices();
            this.wordPos = -1;
            this.currentContexts = new ArrayList<>();
            this.ctxIdx = 0;
            this.needNewWord = true;
            moveToNextWord();
        }

        private void shuffleIndices() {
            for (int i = indices.length - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }

        private void moveToNextWord() {
            while (true) {
                wordPos++;
                if (wordPos >= indices.length) {
                    if (currentEpoch + 1 < epochs) {
                        currentEpoch++;
                        shuffleIndices();
                        wordPos = 0;
                    } else {
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
            }
        }

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
            if (type == Word2VecType.CBOW) {
                ex = generateCbowExample(realPos, centerIdx);
                needNewWord = true;
            } else { // SKIP_GRAMM
                int ctxIdxWord = currentContexts.get(ctxIdx);
                ex = generateSkipGramExample(centerIdx, ctxIdxWord);
                ctxIdx++;
                if (ctxIdx >= currentContexts.size()) needNewWord = true;
            }
            return ex;
        }

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

        private Net.Example generateSkipGramExample(int centerIdx, int ctxIdx) {
            float[] input = new float[vocabSize];
            input[centerIdx] = 1.0f;
            float[] target = new float[vocabSize];
            target[ctxIdx] = 1.0f;
            return new Net.Example(input, target);
        }

        public int totalExamples() {
            if (type == Word2VecType.CBOW) {
                return words.length * epochs;
            } else {
                return words.length * (2 * windowSize) * epochs;
            }
        }
    }

    // ======================== ИТЕРАТОР ДЛЯ SIMPLENET (NEGATIVE SAMPLING) ========================
    public static class ExampleIteratorSimple implements Iterator<SimpleNetWithNegativeSampling.Example> {
        private final String[] words;
        private final int windowSize;
        private final Word2VecType type;
        private final Map<String, Integer> wordToIdx;
        private final int vocabSize;
        private final int negativeCount;
        private final Random rng;
        private final int epochs;
        private int[] indices;
        private int wordPos;
        private int currentEpoch;
        private List<Integer> currentContexts;
        private int ctxIdx;
        private boolean needNewWord;

        public ExampleIteratorSimple(String[] words, int windowSize, Word2VecType type,
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
            this.indices = new int[words.length];
            for (int i = 0; i < indices.length; i++) indices[i] = i;
            shuffleIndices();
            this.wordPos = -1;
            this.currentContexts = new ArrayList<>();
            this.ctxIdx = 0;
            this.needNewWord = true;
            moveToNextWord();
        }

        /** Сброс итератора для следующей эпохи. */
        public void resetEpoch() {
            if (currentEpoch + 1 < epochs) {
                currentEpoch++;
                shuffleIndices();
                wordPos = -1;
                currentContexts.clear();
                ctxIdx = 0;
                needNewWord = true;
                moveToNextWord();
            }
        }

        private void shuffleIndices() {
            for (int i = indices.length - 1; i > 0; i--) {
                int j = rng.nextInt(i + 1);
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }

        private void moveToNextWord() {
            while (true) {
                wordPos++;
                if (wordPos >= indices.length) {
                    // Конец текущей эпохи, дальнейшие вызовы hasNext() вернут false
                    needNewWord = false;
                    currentContexts.clear();
                    return;
                }
                int realPos = indices[wordPos];
                currentContexts = getContextsForWord(realPos);
                if (!currentContexts.isEmpty()) {
                    ctxIdx = 0;
                    needNewWord = false;
                    return;
                }
            }
        }

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
            if (wordPos < indices.length - 1) {
                moveToNextWord();
                return !needNewWord;
            }
            return false;
        }

        @Override
        public SimpleNetWithNegativeSampling.Example next() {
            if (!hasNext()) throw new NoSuchElementException();
            int realPos = indices[wordPos];
            int centerIdx = wordToIdx.get(words[realPos]);

            SimpleNetWithNegativeSampling.Example ex;
            if (type == Word2VecType.CBOW_NEGATIVE_SAMPLING) {
                ex = generateCbowNegativeExample(realPos, centerIdx);
                needNewWord = true;
            } else { // SKIPGRAM_NEGATIVE_SAMPLING
                int ctxIdxWord = currentContexts.get(ctxIdx);
                ex = generateSkipGramNegativeExample(realPos, centerIdx, ctxIdxWord);
                ctxIdx++;
                if (ctxIdx >= currentContexts.size()) needNewWord = true;
            }
            return ex;
        }

        private SimpleNetWithNegativeSampling.Example generateCbowNegativeExample(int pos, int centerIdx) {
            int start = Math.max(0, pos - windowSize);
            int end = Math.min(words.length - 1, pos + windowSize);
            Set<Integer> contextIndices = new HashSet<>();
            for (int j = start; j <= end; j++) {
                if (j == pos) continue;
                contextIndices.add(wordToIdx.get(words[j]));
            }
            if (contextIndices.isEmpty()) {
                // Не должно произойти, так как контекст непустой
                return new SimpleNetWithNegativeSampling.Example(new float[vocabSize], 0, new int[0]);
            }
            float[] input = new float[vocabSize];
            for (int ctx : contextIndices) input[ctx] += 1.0f;
            for (int k = 0; k < vocabSize; k++) input[k] /= contextIndices.size();

            int positive = centerIdx;
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
            return new SimpleNetWithNegativeSampling.Example(input, positive, negatives);
        }

        private SimpleNetWithNegativeSampling.Example generateSkipGramNegativeExample(int pos, int centerIdx, int ctxIdx) {
            float[] input = new float[vocabSize];
            input[centerIdx] = 1.0f;
            int positive = ctxIdx;
            Set<Integer> allContexts = new HashSet<>(currentContexts);
            int[] negatives = new int[negativeCount];
            for (int n = 0; n < negativeCount; n++) {
                int neg;
                do {
                    neg = rng.nextInt(vocabSize);
                } while (neg == centerIdx || allContexts.contains(neg));
                negatives[n] = neg;
                allContexts.add(neg);
            }
            return new SimpleNetWithNegativeSampling.Example(input, positive, negatives);
        }
    }
}