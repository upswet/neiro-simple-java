package neiro.simple.net.loss;

import neiro.simple.net.Tensor;

/**
 * Среднеквадратичная ошибка (Mean Squared Error) для задач регрессии.
 * loss = (1/N) * Σ (pred_i - target_i)²
 */
public class MSELoss implements Loss {

    @Override
    public float compute(Tensor predicted, Tensor target) {
        if (predicted.size != target.size)
            throw new IllegalArgumentException("Size mismatch");
        float sum = 0f;
        for (int i = 0; i < predicted.size; i++) {
            float diff = predicted.data[i] - target.data[i];
            sum += diff * diff;
        }
        return sum / predicted.size;
    }

    @Override
    public Tensor gradient(Tensor predicted, Tensor target) {
        Tensor grad = new Tensor(predicted.shape);
        float scale = 2f / predicted.size;
        for (int i = 0; i < grad.size; i++) {
            grad.data[i] = scale * (predicted.data[i] - target.data[i]);
        }
        return grad;
    }
}
