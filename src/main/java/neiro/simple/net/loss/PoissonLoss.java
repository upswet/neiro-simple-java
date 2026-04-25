package neiro.simple.net.loss;

import neiro.simple.net.Tensor;

/**
 * Poisson loss (negative log-likelihood) для регрессии счётных данных.
 * <p>
 * Модель должна предсказывать интенсивность (rate) – положительное число,
 * например, путём применения exp-активации на последнем слое.
 * loss = (1/N) * Σ (pred_i - y_i * log(pred_i))
 */
public class PoissonLoss implements Loss {
    private static final float EPS = 1e-7f;

    @Override
    public float compute(Tensor predicted, Tensor target) {
        if (predicted.size != target.size)
            throw new IllegalArgumentException("Size mismatch");
        float sum = 0f;
        for (int i = 0; i < predicted.size; i++) {
            float pred = Math.max(predicted.data[i], EPS);
            float t = target.data[i];
            sum += pred - t * (float) Math.log(pred);
        }
        return sum / predicted.size;
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor target) {
        Tensor grad = new Tensor(predicted.shape);
        float scale = 1f / predicted.size;
        for (int i = 0; i < grad.size; i++) {
            float pred = Math.max(predicted.data[i], EPS);
            float t = target.data[i];
            grad.data[i] = scale * (1f - t / pred);
        }
        return grad;
    }
}