package neiro.simple.net.metric;

import neiro.simple.net.Tensor;

/**
 * Точность (Accuracy) для многоклассовой классификации.
 * Поддерживает как one-hot метки (Tensor), так и индексы классов (int[]).
 * Работает как с батчами, так и с одиночными примерами.
 */
public class CategoricalAccuracyMetric implements Metric {
    private int correct;
    private int total;

    @Override
    public void update(Tensor predicted, Object target) {
        // Приводим predicted к 2D виду [batchSize, numClasses]
        Tensor pred2D;
        int batchSize;
        int numClasses;
        if (predicted.rank == 1) {
            // Одиночный пример: [numClasses] -> [1, numClasses]
            batchSize = 1;
            numClasses = predicted.shape[0];
            pred2D = predicted.reshape(1, numClasses);
        } else if (predicted.rank == 2) {
            // Батч: [batchSize, numClasses]
            batchSize = predicted.shape[0];
            numClasses = predicted.shape[1];
            pred2D = predicted;
        } else {
            throw new IllegalArgumentException("Predicted tensor must be 1D or 2D");
        }

        if (target instanceof Tensor) {
            Tensor targ = (Tensor) target;
            Tensor targ2D;
            if (targ.rank == 1) {
                targ2D = targ.reshape(1, numClasses);
            } else if (targ.rank == 2) {
                targ2D = targ;
            } else {
                throw new IllegalArgumentException("Target tensor must be 1D or 2D");
            }
            for (int i = 0; i < batchSize; i++) {
                int predIdx = argmax(pred2D.data, i * numClasses, numClasses);
                int targIdx = argmax(targ2D.data, i * numClasses, numClasses);
                if (predIdx == targIdx) correct++;
                total++;
            }
        } else if (target instanceof int[]) {
            int[] labels = (int[]) target;
            if (labels.length != batchSize) {
                throw new IllegalArgumentException("Target array length must match batch size");
            }
            for (int i = 0; i < batchSize; i++) {
                int predIdx = argmax(pred2D.data, i * numClasses, numClasses);
                if (predIdx == labels[i]) correct++;
                total++;
            }
        } else {
            throw new IllegalArgumentException("Target must be Tensor or int[]");
        }
    }

    private int argmax(float[] data, int start, int len) {
        int best = 0;
        float max = data[start];
        for (int i = 1; i < len; i++) {
            float val = data[start + i];
            if (val > max) {
                max = val;
                best = i;
            }
        }
        return best;
    }

    @Override
    public void reset() {
        correct = 0;
        total = 0;
    }

    @Override
    public float getValue() {
        return total == 0 ? 0.0f : (float) correct / total;
    }

    @Override
    public String name() {
        return "accuracy";
    }
}