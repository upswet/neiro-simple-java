package neiro.simple.net.loss;

import neiro.simple.net.Tensor;
import java.util.Random;
import java.util.HashSet;

/**
 * Negative Sampling Loss для задач с большим количеством классов.
 * <p>
 * Для каждого примера из батча:
 * <ul>
 *   <li>берётся логит правильного класса (positive sample, метка = 1)</li>
 *   <li>сэмплируется {@code numNegativeSamples} случайных классов, не совпадающих с правильным
 *       (negative samples, метка = 0)</li>
 *   <li>вычисляется сумма бинарных кросс‑энтропий:
 *       loss = - [ log(sigmoid(score_pos)) + Σ log(1 - sigmoid(score_neg_i)) ]</li>
 * </ul>
 * Градиент вычисляется только для использованных логитов, остальные остаются нулевыми.
 * </p>
 * <p>
 * Для сэмплирования негативных примеров можно задать веса (например, частоты слов).
 * По умолчанию используется равномерное распределение.
 * </p>
 *
 *
 *
 * Как использовать
 * java
 * // Модель должна заканчиваться Dense слоем без активации (логиты)
 * Model model = new Model()
 *     .addLayer(new DenseLayer(inputSize, hiddenSize))
 *     .addLayer(new ReLULayer())
 *     .addLayer(new DenseLayer(hiddenSize, numClasses))   // выход: логиты
 *     .initializeParameters();
 *
 * // Используем NegativeSamplingLoss
 * Loss loss = new NegativeSamplingLoss(numClasses, 5); // 5 негативных примеров
 *
 * // Trainer работает как обычно
 * Trainer trainer = new Trainer();
 * trainer.fit(model, optimizer, loss, trainLoader, callback, testLoader, batchSize, epochs);
 *
 * Для больших словарей (миллионы классов) этот подход всё ещё не оптимален, так как модель вычисляет логиты для всех классов.
 */
public class NegativeSamplingLoss implements Loss {

    private final int numClasses;
    private final int numNegativeSamples;
    private final float[] sampleProbs; // вероятности для сэмплирования (null = равномерно)
    private final Random random;

    // Для backward нужно запомнить, какие индексы были использованы
    private int[] lastPosIndices;   // [batchSize] – правильные классы
    private int[][] lastNegIndices; // [batchSize][numNegativeSamples] – негативные классы

    /**
     * Конструктор с равномерным сэмплированием негативных классов.
     *
     * @param numClasses          общее количество классов
     * @param numNegativeSamples  сколько негативных примеров генерировать на один положительный
     */
    public NegativeSamplingLoss(int numClasses, int numNegativeSamples) {
        this(numClasses, numNegativeSamples, null);
    }

    /**
     * Конструктор с заданным распределением для сэмплирования.
     *
     * @param numClasses          общее количество классов
     * @param numNegativeSamples  сколько негативных примеров генерировать
     * @param sampleProbs         вероятности выбора каждого класса (длина = numClasses).
     *                            Если null, используется равномерное распределение.
     */
    public NegativeSamplingLoss(int numClasses, int numNegativeSamples, float[] sampleProbs) {
        this.numClasses = numClasses;
        this.numNegativeSamples = numNegativeSamples;
        this.random = new Random();

        if (sampleProbs != null) {
            if (sampleProbs.length != numClasses) {
                throw new IllegalArgumentException("sampleProbs length must be " + numClasses);
            }
            // Нормализуем, чтобы сумма была 1
            float sum = 0f;
            for (float p : sampleProbs) sum += p;
            this.sampleProbs = new float[numClasses];
            for (int i = 0; i < numClasses; i++) {
                this.sampleProbs[i] = sampleProbs[i] / sum;
            }
        } else {
            this.sampleProbs = null;
        }
    }

    @Override
    public float compute(Tensor predicted, Tensor target) {
        // Приводим predicted к 2D [batchSize, numClasses]
        int batchSize = predicted.size / predicted.shape[predicted.rank - 1];
        int nClasses = predicted.shape[predicted.rank - 1];
        if (nClasses != numClasses) {
            throw new IllegalArgumentException("Predicted last dimension must be " + numClasses);
        }
        Tensor pred2D = predicted.reshape(batchSize, numClasses);

        // Извлекаем индексы правильных классов
        int[] posIndices = extractPositiveIndices(target, batchSize);
        lastPosIndices = posIndices.clone();
        lastNegIndices = new int[batchSize][numNegativeSamples];

        float totalLoss = 0f;

        for (int i = 0; i < batchSize; i++) {
            int posIdx = posIndices[i];
            float posLogit = pred2D.get(i, posIdx);

            // Сэмплируем негативные классы
            HashSet<Integer> used = new HashSet<>();
            used.add(posIdx);
            int[] negIdx = new int[numNegativeSamples];
            for (int s = 0; s < numNegativeSamples; s++) {
                int sample;
                do {
                    sample = sampleClass();
                } while (used.contains(sample));
                used.add(sample);
                negIdx[s] = sample;
            }
            lastNegIndices[i] = negIdx;

            // Вычисляем loss для данного примера
            // loss_pos = -log(sigmoid(posLogit))
            float sigPos = sigmoid(posLogit);
            float lossPos = - (float) Math.log(Math.max(sigPos, 1e-7f));

            float lossNeg = 0f;
            for (int s = 0; s < numNegativeSamples; s++) {
                float negLogit = pred2D.get(i, negIdx[s]);
                float sigNeg = sigmoid(negLogit);
                // -log(1 - sigmoid(negLogit)) = -log(sigmoid(-negLogit))
                lossNeg += - (float) Math.log(Math.max(1f - sigNeg, 1e-7f));
            }

            totalLoss += lossPos + lossNeg;
        }

        return totalLoss / batchSize; // усредняем по батчу
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor target) {
        int batchSize = predicted.size / predicted.shape[predicted.rank - 1];
        Tensor pred2D = predicted.reshape(batchSize, numClasses);

        // Градиент по логитам: для положительного класса = sigmoid(logit) - 1,
        // для отрицательных = sigmoid(logit).
        // Все остальные логиты получают 0 градиента.
        Tensor grad = new Tensor(predicted.shape); // заполнен нулями

        for (int i = 0; i < batchSize; i++) {
            int posIdx = lastPosIndices[i];
            float posLogit = pred2D.get(i, posIdx);
            float sigPos = sigmoid(posLogit);
            grad.set(sigPos - 1f, i, posIdx);  // производная -log(sigmoid) = sigmoid - 1

            for (int s = 0; s < numNegativeSamples; s++) {
                int negIdx = lastNegIndices[i][s];
                float negLogit = pred2D.get(i, negIdx);
                float sigNeg = sigmoid(negLogit);
                grad.set(sigNeg, i, negIdx);   // производная -log(1-sigmoid) = sigmoid
            }
        }

        // Делим на batchSize для согласованности с compute (опционально)
        // В Trainer градиент потом используется как есть, оптимизатор сам делит при необходимости.
        // Здесь не делим, т.к. обычно loss уже средний, и градиент должен быть средним.
        // Но если хотим, чтобы шаг оптимизатора не зависел от размера батча, нужно делить.
        // В CategoricalCrossEntropyLoss не делили, оставим так же.
        return grad;
    }

    /**
     * Извлекает индексы правильных классов из target.
     * Поддерживаются:
     * <ul>
     *   <li>Tensor one-hot (1D или 2D)</li>
     *   <li>int[] – массив индексов</li>
     * </ul>
     */
    private int[] extractPositiveIndices(Tensor target, int batchSize) {
        int[] indices = new int[batchSize];
        if (target.rank == 1) {
            // один one-hot вектор
            if (batchSize != 1) throw new IllegalArgumentException("Batch size mismatch");
            indices[0] = argmax(target.data, 0, target.size);
        } else if (target.rank == 2) {
            // батч one-hot
            if (target.shape[0] != batchSize) throw new IllegalArgumentException("Batch size mismatch");
            int nClasses = target.shape[1];
            for (int i = 0; i < batchSize; i++) {
                indices[i] = argmax(target.data, i * nClasses, nClasses);
            }
        } else {
            throw new IllegalArgumentException("Target tensor must be 1D or 2D one-hot");
        }
        return indices;
    }

    private int[] extractPositiveIndices(Object target, int batchSize) {
        if (target instanceof Tensor) {
            return extractPositiveIndices((Tensor) target, batchSize);
        } else if (target instanceof int[]) {
            int[] arr = (int[]) target;
            if (arr.length != batchSize) {
                throw new IllegalArgumentException("Target array length must match batch size");
            }
            return arr.clone();
        } else {
            throw new IllegalArgumentException("Target must be Tensor or int[]");
        }
    }

    private int argmax(float[] data, int start, int len) {
        int best = 0;
        float max = data[start];
        for (int i = 1; i < len; i++) {
            if (data[start + i] > max) {
                max = data[start + i];
                best = i;
            }
        }
        return best;
    }

    private float sigmoid(float x) {
        if (x >= 0) {
            return 1f / (1f + (float) Math.exp(-x));
        } else {
            float expX = (float) Math.exp(x);
            return expX / (1f + expX);
        }
    }

    /**
     * Сэмплирует класс согласно заданному распределению.
     */
    private int sampleClass() {
        if (sampleProbs == null) {
            return random.nextInt(numClasses);
        } else {
            float r = random.nextFloat();
            float cum = 0f;
            for (int i = 0; i < numClasses; i++) {
                cum += sampleProbs[i];
                if (r < cum) return i;
            }
            return numClasses - 1;
        }
    }
}
