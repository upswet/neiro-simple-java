package neiro.simple.net.loss;

import neiro.simple.net.Tensor;

/**
 * Робастная функция потерь для регрессии, сочетающая MSE и MAE.
 * Huber Loss для регрессии с выбросами.
 * При |error| <= delta: 0.5 * error²
 * При |error| > delta:  delta * |error| - 0.5 * delta²
 * loss = среднее по всем элементам.
 */
public class HuberLoss implements Loss {
    private final float delta;

    public HuberLoss(float delta) {
        this.delta = delta;
    }

    public HuberLoss() {
        this(1.0f);
    }

    @Override
    public float compute(Tensor predicted, Tensor target) {
        if (predicted.size != target.size)
            throw new IllegalArgumentException("Size mismatch");
        float sum = 0f;
        for (int i = 0; i < predicted.size; i++) {
            float error = predicted.data[i] - target.data[i];
            float absError = Math.abs(error);
            if (absError <= delta) {
                sum += 0.5f * error * error;
            } else {
                sum += delta * absError - 0.5f * delta * delta;
            }
        }
        return sum / predicted.size;
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor target) {
        Tensor grad = new Tensor(predicted.shape);
        float scale = 1f / predicted.size;
        for (int i = 0; i < grad.size; i++) {
            float error = predicted.data[i] - target.data[i];
            if (Math.abs(error) <= delta) {
                grad.data[i] = scale * error;
            } else {
                grad.data[i] = scale * delta * Math.signum(error);
            }
        }
        return grad;
    }
}