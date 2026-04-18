package neiro.simple.net.layer;

import neiro.simple.net.Tensor;

/**
 * Преобразует входной тензор в плоский вектор или матрицу.
 * <ul>
 *   <li>1D → 1D (без изменений)</li>
 *   <li>2D → 2D (предполагается [batch, features])</li>
 *   <li>3D → 1D (один пример: [H, W, C] → [H*W*C])</li>
 *   <li>4D → 2D (батч: [batch, H, W, C] → [batch, H*W*C])</li>
 * </ul>
 */
public class FlattenLayer implements Layer {
    private int[] inputShape;
    private int inputRank;

    @Override
    public void initializeParameters(Layer prev, Layer next) {}

    @Override
    public Tensor forward(Tensor input) {
        inputShape = input.shape.clone();
        inputRank = input.rank;

        switch (input.rank) {
            case 1:
                // Уже плоский вектор
                return input;
            case 2:
                // [batch, features] – оставляем как есть
                return input;
            case 3:
                // [H, W, C] -> [H*W*C]
                int features3D = input.shape[0] * input.shape[1] * input.shape[2];
                return input.reshape(features3D);
            case 4:
                // [batch, H, W, C] -> [batch, H*W*C]
                int batchSize = input.shape[0];
                int features4D = input.shape[1] * input.shape[2] * input.shape[3];
                return input.reshape(batchSize, features4D);
            default:
                throw new IllegalArgumentException("FlattenLayer: unsupported rank " + input.rank);
        }
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // Восстанавливаем исходную форму, запомненную в forward
        if (inputRank == 1 || inputRank == 2) {
            return gradOutput.reshape(inputShape);
        } else if (inputRank == 3) {
            // gradOutput имеет форму [features] -> [H, W, C]
            return gradOutput.reshape(inputShape);
        } else if (inputRank == 4) {
            // gradOutput имеет форму [batch, features] -> [batch, H, W, C]
            return gradOutput.reshape(inputShape);
        } else {
            throw new IllegalStateException("Unknown input rank in backward");
        }
    }
}