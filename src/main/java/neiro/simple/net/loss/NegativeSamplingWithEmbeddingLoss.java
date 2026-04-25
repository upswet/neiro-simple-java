package neiro.simple.net.loss;

import neiro.simple.net.Parameter;
import neiro.simple.net.Tensor;
import java.util.*;

/**
 * NegativeSamplingWithEmbeddingLoss (уже есть)
 * Когда использовать: очень большое количество классов (сотни тысяч и миллионы), когда полный softmax невозможен. Классический пример – word2vec, рекомендательные системы.
 *
 * Требование: модель возвращает эмбеддинг (скрытый вектор), а loss содержит матрицу эмбеддингов классов.
 *
 *
 * Negative Sampling Loss с использованием эмбеддингов выходного слоя.
 * <p>
 * Модель должна возвращать скрытый вектор (hidden) формы [batchSize, hiddenDim].
 * Loss содержит обучаемую матрицу эмбеддингов классов размером [numClasses, hiddenDim]
 * и опциональные смещения (bias) для каждого класса.
 * </p>
 * <p>
 * Для каждого примера loss = -log(sigmoid(hidden·emb_pos + bias_pos))
 *                           - Σ log(sigmoid(-hidden·emb_neg_i - bias_neg_i))
 * </p>
 * <p>
 * Градиенты распространяются как на скрытый вектор (возвращается в модель),
 * так и на эмбеддинги затронутых классов.
 * </p>
 *
 * Для задач с очень большим количеством классов (например, в NLP или рекомендательных системах) вычисление полного Softmax становится узким местом. Negative sampling решает эту проблему, заменяя многоклассовую классификацию на серию бинарных задач: модель учится отличать правильный класс от нескольких случайно выбранных «негативных».
 *
 * В предыдущем ответе была представлена базовая версия NegativeSamplingLoss, которая всё ещё требовала вычисления логитов для всех классов.
 * Теперь мы реализуем расширенную версию — NegativeSamplingWithEmbeddingLoss, где модель возвращает только скрытое представление (embedding), а функция потерь содержит собственную обучаемую матрицу эмбеддингов классов. Это классический подход, используемый в word2vec, и он позволяет работать с миллионами классов.
 *
 * // Модель возвращает скрытое представление (без выходного слоя)
 * Model model = new Model()
 *     .addLayer(new DenseLayer(inputSize, 256))
 *     .addLayer(new ReLULayer())
 *     .addLayer(new DenseLayer(256, hiddenDim))   // hiddenDim = 128
 *     .initializeParameters();
 *
 * NegativeSamplingWithEmbeddingLoss loss = new NegativeSamplingWithEmbeddingLoss(
 *     numClasses,      // 100000
 *     hiddenDim,       // 128
 *     5                // 5 негативных примеров
 * );
 *
 * // Важно: добавить параметры loss к параметрам модели перед созданием оптимизатора
 * List<Parameter> allParams = new ArrayList<>();
 * allParams.addAll(model.parameters());
 * allParams.addAll(loss.parameters());
 *
 * Optimizer optimizer = new Adam(0.001f);
 * Trainer trainer = new Trainer();
 * trainer.addMetric(new CategoricalAccuracyMetric());
 *
 * // В Trainer.fit нужно передать объединённый список параметров.
 * // Для этого можно модифицировать Trainer, либо обернуть параметры в модель-прокси.
 * // Простейший способ – добавить метод Model.addParameters(Collection<Parameter>).
 *
 *
 * NegativeSamplingWithEmbeddingLoss хранит именно выходные эмбеддинги классов. Скрытое представление, генерируемое моделью, — это не эмбеддинг класса, а эмбеддинг конкретного входного примера (например, изображения). Чтобы получить эмбеддинг класса, нужно обратиться к classEmbeddings.
 * После обучения эмбеддинги классов можно извлечь из loss и использовать для анализа или предсказаний.
 */
public class NegativeSamplingWithEmbeddingLoss implements Loss {

    private final int numClasses;
    private final int hiddenDim;
    private final int numNegativeSamples;
    private final float[] sampleProbs; // вероятности сэмплирования (null = равномерно)
    private final Random random;

    // Обучаемые параметры
    private final Parameter classEmbeddings; // [numClasses, hiddenDim]     матрица, где каждая строка — это вектор (эмбеддинг) одного класса
    private final Parameter classBiases;     // [numClasses]    смещение для каждого класса

    // Кэш для backward
    private int[] lastPosIndices;
    private int[][] lastNegIndices;
    private Tensor lastHidden;

    /**
     * @param numClasses         общее количество классов
     * @param hiddenDim          размерность скрытого вектора модели
     * @param numNegativeSamples сколько негативных примеров на один позитивный
     */
    public NegativeSamplingWithEmbeddingLoss(int numClasses, int hiddenDim, int numNegativeSamples) {
        this(numClasses, hiddenDim, numNegativeSamples, null);
    }

    /**
     * @param numClasses         общее количество классов
     * @param hiddenDim          размерность скрытого вектора
     * @param numNegativeSamples количество негативных примеров
     * @param sampleProbs        вероятности выбора классов (null = равномерно)
     */
    public NegativeSamplingWithEmbeddingLoss(int numClasses, int hiddenDim, int numNegativeSamples,
                                             float[] sampleProbs) {
        this.numClasses = numClasses;
        this.hiddenDim = hiddenDim;
        this.numNegativeSamples = numNegativeSamples;
        this.random = new Random();

        if (sampleProbs != null) {
            if (sampleProbs.length != numClasses)
                throw new IllegalArgumentException("sampleProbs length must be " + numClasses);
            float sum = 0f;
            for (float p : sampleProbs) sum += p;
            this.sampleProbs = new float[numClasses];
            for (int i = 0; i < numClasses; i++) this.sampleProbs[i] = sampleProbs[i] / sum;
        } else {
            this.sampleProbs = null;
        }

        // Инициализация эмбеддингов (Xavier)
        this.classEmbeddings = new Parameter(numClasses, hiddenDim);
        this.classEmbeddings.initXavier(true);
        this.classBiases = new Parameter(numClasses);
    }

    /** Возвращает обучаемые параметры этого loss (нужно добавить к оптимизатору). */
    public List<Parameter> parameters() {
        return List.of(classEmbeddings, classBiases);
    }

    @Override
    public float compute(Tensor predicted, Tensor target) {
        if (predicted.rank != 2 || predicted.shape[1] != hiddenDim)
            throw new IllegalArgumentException("Predicted must be [batchSize, hiddenDim]");

        int batchSize = predicted.shape[0];
        lastHidden = predicted;

        int[] posIndices = extractPositiveIndices(target, batchSize);
        lastPosIndices = posIndices.clone();
        lastNegIndices = new int[batchSize][numNegativeSamples];

        float totalLoss = 0f;

        for (int i = 0; i < batchSize; i++) {
            int posIdx = posIndices[i];
            float posScore = dotWithEmbedding(predicted, i, posIdx) + classBiases.data.get(posIdx);
            float sigPos = sigmoid(posScore);
            float lossPos = - (float) Math.log(Math.max(sigPos, 1e-7f));

            // Сэмплируем негативные классы (без повторений)
            Set<Integer> used = new HashSet<>();
            used.add(posIdx);
            int[] negIdx = new int[numNegativeSamples];
            for (int s = 0; s < numNegativeSamples; s++) {
                int sample;
                do { sample = sampleClass(); } while (used.contains(sample));
                used.add(sample);
                negIdx[s] = sample;
            }
            lastNegIndices[i] = negIdx;

            float lossNeg = 0f;
            for (int s = 0; s < numNegativeSamples; s++) {
                int neg = negIdx[s];
                float negScore = dotWithEmbedding(predicted, i, neg) + classBiases.data.get(neg);
                float sigNeg = sigmoid(negScore);
                lossNeg += - (float) Math.log(Math.max(1f - sigNeg, 1e-7f));
            }
            totalLoss += lossPos + lossNeg;
        }
        return totalLoss / batchSize;
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor target) {
        int batchSize = predicted.shape[0];
        Tensor gradHidden = new Tensor(batchSize, hiddenDim); // нули

        for (int i = 0; i < batchSize; i++) {
            int posIdx = lastPosIndices[i];

            // 1. Позитивный пример
            float posScore = dotWithEmbedding(lastHidden, i, posIdx) + classBiases.data.get(posIdx);
            float gradPosScore = sigmoid(posScore) - 1f; // производная -log(sigmoid)
            addScaledEmbeddingToGrad(gradHidden, i, posIdx, gradPosScore);
            addScaledHiddenToEmbeddingGrad(lastHidden, i, posIdx, gradPosScore);
            classBiases.grad.data[posIdx] += gradPosScore;

            // 2. Негативные примеры
            for (int s = 0; s < numNegativeSamples; s++) {
                int negIdx = lastNegIndices[i][s];
                float negScore = dotWithEmbedding(lastHidden, i, negIdx) + classBiases.data.get(negIdx);
                float gradNegScore = sigmoid(negScore); // производная -log(1-sigmoid)
                addScaledEmbeddingToGrad(gradHidden, i, negIdx, gradNegScore);
                addScaledHiddenToEmbeddingGrad(lastHidden, i, negIdx, gradNegScore);
                classBiases.grad.data[negIdx] += gradNegScore;
            }
        }

        // Усреднение по батчу
        float scale = 1.0f / batchSize;
        for (int i = 0; i < gradHidden.size; i++) gradHidden.data[i] *= scale;
        for (int i = 0; i < classEmbeddings.grad.size; i++) classEmbeddings.grad.data[i] *= scale;
        for (int i = 0; i < classBiases.grad.size; i++) classBiases.grad.data[i] *= scale;

        return gradHidden;
    }

    // === Вспомогательные методы ===
    private float dotWithEmbedding(Tensor hidden, int row, int classIdx) {
        int offH = row * hiddenDim;
        int offE = classIdx * hiddenDim;
        float sum = 0f;
        for (int d = 0; d < hiddenDim; d++)
            sum += hidden.data[offH + d] * classEmbeddings.data.data[offE + d];
        return sum;
    }

    private void addScaledEmbeddingToGrad(Tensor gradHidden, int row, int classIdx, float scale) {
        int offH = row * hiddenDim;
        int offE = classIdx * hiddenDim;
        for (int d = 0; d < hiddenDim; d++)
            gradHidden.data[offH + d] += scale * classEmbeddings.data.data[offE + d];
    }

    private void addScaledHiddenToEmbeddingGrad(Tensor hidden, int row, int classIdx, float scale) {
        int offH = row * hiddenDim;
        int offE = classIdx * hiddenDim;
        float[] embGrad = classEmbeddings.grad.data;
        for (int d = 0; d < hiddenDim; d++)
            embGrad[offE + d] += scale * hidden.data[offH + d];
    }

    private int[] extractPositiveIndices(Object target, int batchSize) {
        if (target instanceof Tensor) {
            Tensor t = (Tensor) target;
            if (t.rank == 1) {
                if (batchSize != 1) throw new IllegalArgumentException("Batch size mismatch");
                return new int[]{argmax(t.data, 0, t.size)};
            } else if (t.rank == 2) {
                if (t.shape[0] != batchSize) throw new IllegalArgumentException("Batch size mismatch");
                int n = t.shape[1];
                int[] idx = new int[batchSize];
                for (int i = 0; i < batchSize; i++) idx[i] = argmax(t.data, i * n, n);
                return idx;
            }
        } else if (target instanceof int[]) {
            int[] arr = (int[]) target;
            if (arr.length != batchSize) throw new IllegalArgumentException("Target array length mismatch");
            return arr.clone();
        }
        throw new IllegalArgumentException("Target must be Tensor or int[]");
    }

    public static int argmax(float[] data, int start, int len) {
        int best = 0;
        float max = data[start];
        for (int i = 1; i < len; i++)
            if (data[start + i] > max) { max = data[start + i]; best = i; }
        return best;
    }

    private float sigmoid(float x) {
        return x >= 0 ? 1f / (1f + (float) Math.exp(-x))
                : (float) Math.exp(x) / (1f + (float) Math.exp(x));
    }

    private int sampleClass() {
        if (sampleProbs == null) return random.nextInt(numClasses);
        float r = random.nextFloat();
        float cum = 0f;
        for (int i = 0; i < numClasses; i++) {
            cum += sampleProbs[i];
            if (r < cum) return i;
        }
        return numClasses - 1;
    }

    /**
     * Вычисляет логит (score) для заданного класса на основе скрытого вектора.
     * Используется для оценки точности, когда нужно перебрать все классы.
     *
     * @param hidden   скрытый вектор батча [batchSize, hiddenDim]
     * @param row      индекс примера в батче
     * @param classIdx индекс класса
     * @return score = hidden·embedding[classIdx] + bias[classIdx]
     */
    public float computeScoreForClass(Tensor hidden, int row, int classIdx) {
        return dotWithEmbedding(hidden, row, classIdx) + classBiases.data.get(classIdx);
    }
}