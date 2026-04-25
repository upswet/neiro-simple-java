package neiro.simple.net.loss;

import neiro.simple.net.Tensor;

/**
 Бинарная кросс-энтропия (BinaryCrossEntropyLoss)
 Когда использовать: бинарная классификация или многоклассовая с мульти-лейблами (например, наличие нескольких тегов у изображения). Последний слой – Sigmoid.
 */
public class BinaryCrossEntropyLoss implements Loss {
    private static final float EPS = 1e-7f;

    @Override
    public float compute(Tensor predicted, Tensor target) {
        if (predicted.size != target.size) throw new IllegalArgumentException("Size mismatch");
        float sum = 0.0f;
        for (int i = 0; i < predicted.size; i++) {
            float p = Math.max(EPS, Math.min(1.0f - EPS, predicted.data[i]));
            float y = target.data[i];
            sum += -y * Math.log(p) - (1.0f - y) * Math.log(1.0f - p);
        }
        return sum / predicted.size;
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor target) {
        Tensor grad = new Tensor(predicted.shape);
        for (int i = 0; i < grad.size; i++) {
            float p = Math.max(EPS, Math.min(1.0f - EPS, predicted.data[i]));
            float y = target.data[i];
            grad.data[i] = (p - y) / (p * (1.0f - p) + EPS);
        }
        return grad;
    }
}
