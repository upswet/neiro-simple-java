package neiro.simple.net.metric;

import neiro.simple.net.Tensor;
import java.util.HashMap;
import java.util.Map;

/**
 * Метрики Precision, Recall, F1 (macro average) для многоклассовой классификации.
 * Поддерживает как one-hot метки (Tensor), так и индексы классов (int[]).
 */
public class F1Metric implements Metric {
    private final Map<Integer, ConfusionStats> stats = new HashMap<>();
    private int numClasses = -1;

    private static class ConfusionStats {
        int tp = 0, fp = 0, fn = 0;
    }

    @Override
    public void update(Tensor predicted, Object target) {
        // Приводим predicted к 2D виду [batchSize, numClasses]
        Tensor pred2D;
        int batchSize;
        int nClasses;
        if (predicted.rank == 1) {
            batchSize = 1;
            nClasses = predicted.shape[0];
            pred2D = predicted.reshape(1, nClasses);
        } else if (predicted.rank == 2) {
            batchSize = predicted.shape[0];
            nClasses = predicted.shape[1];
            pred2D = predicted;
        } else {
            throw new IllegalArgumentException("Predicted must be 1D or 2D");
        }

        if (numClasses == -1) numClasses = nClasses;
        else if (numClasses != nClasses) throw new IllegalStateException("Class count mismatch");

        int[] trueLabels = getTrueLabels(target, batchSize, nClasses);
        int[] predLabels = getPredLabels(pred2D, batchSize);

        for (int i = 0; i < batchSize; i++) {
            int trueLabel = trueLabels[i];
            int predLabel = predLabels[i];

            stats.computeIfAbsent(trueLabel, k -> new ConfusionStats());
            stats.computeIfAbsent(predLabel, k -> new ConfusionStats());

            if (predLabel == trueLabel) {
                stats.get(trueLabel).tp++;
            } else {
                stats.get(trueLabel).fn++;
                stats.get(predLabel).fp++;
            }
        }
    }

    private int[] getTrueLabels(Object target, int batchSize, int numClasses) {
        if (target instanceof Tensor) {
            Tensor targ = (Tensor) target;
            Tensor targ2D = targ.reshape(batchSize, numClasses);
            int[] labels = new int[batchSize];
            for (int i = 0; i < batchSize; i++) {
                labels[i] = argmax(targ2D.data, i * numClasses, numClasses);
            }
            return labels;
        } else if (target instanceof int[]) {
            return (int[]) target;
        } else {
            throw new IllegalArgumentException("Target must be Tensor or int[]");
        }
    }

    private int[] getPredLabels(Tensor pred2D, int batchSize) {
        int numClasses = pred2D.shape[1];
        int[] labels = new int[batchSize];
        for (int i = 0; i < batchSize; i++) {
            labels[i] = argmax(pred2D.data, i * numClasses, numClasses);
        }
        return labels;
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

    @Override
    public void reset() {
        stats.clear();
        numClasses = -1;
    }

    @Override
    public float getValue() {
        if (stats.isEmpty()) return 0f;
        double totalF1 = 0.0;
        int classesWithData = 0;
        for (Map.Entry<Integer, ConfusionStats> e : stats.entrySet()) {
            ConfusionStats s = e.getValue();
            int tp = s.tp, fp = s.fp, fn = s.fn;
            double precision = (tp + fp) == 0 ? 0 : (double) tp / (tp + fp);
            double recall = (tp + fn) == 0 ? 0 : (double) tp / (tp + fn);
            double f1 = (precision + recall) == 0 ? 0 : 2 * precision * recall / (precision + recall);
            totalF1 += f1;
            classesWithData++;
        }
        return (float) (totalF1 / classesWithData);
    }

    @Override
    public String name() {
        return "f1_macro";
    }
}