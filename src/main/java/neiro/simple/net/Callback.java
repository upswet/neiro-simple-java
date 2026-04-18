package neiro.simple.net;

import java.util.Map;

/**
 * Колбэк для отслеживания процесса обучения.
 */
public interface Callback {
    default void onTrainBegin(Trainer trainer) {}
    default void onTrainEnd(Trainer trainer) {}
    default void onEpochBegin(Trainer trainer, int epoch) {}
    default void onEpochEnd(Trainer trainer, int epoch, Map<String, Float> logs) {}
    default void onBatchBegin(Trainer trainer, int epoch, int batch) {}
    default void onBatchEnd(Trainer trainer, int epoch, int batch, float loss) {}
}
