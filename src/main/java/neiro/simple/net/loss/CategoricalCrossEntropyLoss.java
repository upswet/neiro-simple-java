package neiro.simple.net.loss;

import neiro.simple.net.Tensor;

/**
 * Категориальная кросс-энтропия для one-hot меток.
 * Ожидается, что predicted уже прошёл через Softmax.
 */
public class CategoricalCrossEntropyLoss implements Loss {
    private static final float EPS = 1e-7f;

    @Override
    public float compute(Tensor predicted, Tensor target) {
        // predicted и target имеют одинаковую форму, последнее измерение = число классов
        int batchSize = predicted.size / predicted.shape[predicted.rank - 1];
        int numClasses = predicted.shape[predicted.rank - 1];
        Tensor pred2D = predicted.reshape(batchSize, numClasses);
        Tensor targ2D = target.reshape(batchSize, numClasses);

        float sum = 0.0f;
        for (int i = 0; i < pred2D.size; i++) {
            float p = Math.max(EPS, Math.min(1.0f - EPS, pred2D.data[i]));
            sum -= targ2D.data[i] * Math.log(p);
        }
        return sum / batchSize;
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor target) {
        // Для Softmax + CCE градиент упрощается до (p - y)
        // (предполагается, что backward вызывается сразу после compute на том же predicted)
        Tensor grad = predicted.copy();
        for (int i = 0; i < grad.size; i++) {
            grad.data[i] -= target.data[i];
        }
        // Делить на batchSize не нужно, т.к. усреднение обычно происходит в оптимизаторе
        // или можно разделить, но тогда loss должен быть суммой, а не средним.
        // Оставим как (p - y) – это общепринято.
        return grad;
    }
}
