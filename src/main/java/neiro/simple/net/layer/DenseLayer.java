package neiro.simple.net.layer;

import neiro.simple.net.Parameter;
import neiro.simple.net.layer.fun.*;
import neiro.simple.net.Tensor;

import java.util.List;

/**
 * Полносвязный слой (Dense / Fully Connected).
 * <p>
 * Вход: тензор произвольной формы, последнее измерение = inFeatures.<br>
 * Выход: тензор той же формы, но последнее измерение заменено на outFeatures.
 */
public class DenseLayer implements Layer {
    private final int inFeatures;
    private final int outFeatures;
    private Parameter weights; // [inFeatures, outFeatures]
    private Parameter bias;    // [outFeatures]

    private Tensor lastInput;  // для backward

    public DenseLayer(int inFeatures, int outFeatures) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
    }

    @Override
    public void initializeParameters(Layer prev, Layer next) {
        weights = new Parameter(inFeatures, outFeatures);
        bias = new Parameter(outFeatures);

        if (next instanceof ReLULayer || next instanceof LeakyReLULayer) {
            weights.initHe(true);
        } else if (next instanceof SigmoidLayer || next instanceof TanhLayer) {
            weights.initXavier(true);
        } else if (next instanceof SoftmaxLayer) {
            weights.initXavier(true);
        } else {
            weights.initXavier(true);
        }
        // bias остаётся 0
    }

    @Override
    public Tensor forward(Tensor input) {
        // Проверяем, что последнее измерение совпадает с inFeatures
        if (input.shape[input.rank - 1] != inFeatures) {
            throw new IllegalArgumentException("Input last dim must be " + inFeatures);
        }
        lastInput = input;

        // Вычисляем размер батча (произведение всех измерений, кроме последнего)
        int batchSize = input.size / inFeatures;

        // Преобразуем вход в матрицу [batchSize, inFeatures]
        Tensor inputMat = input.reshape(batchSize, inFeatures);
        Tensor weightsMat = weights.data.reshape(inFeatures, outFeatures);

        // Умножение
        Tensor outMat = Tensor.matmul(inputMat, weightsMat); // [batchSize, outFeatures]

        // Добавляем bias
        float[] b = bias.data.data;
        for (int i = 0; i < batchSize; i++) {
            int off = i * outFeatures;
            for (int j = 0; j < outFeatures; j++) {
                outMat.data[off + j] += b[j];
            }
        }

        // Восстанавливаем исходную форму с новым последним измерением
        int[] outShape = input.shape.clone();
        outShape[outShape.length - 1] = outFeatures;
        return outMat.reshape(outShape);
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        int batchSize = lastInput.size / inFeatures;

        // Градиент выхода -> матрица [batchSize, outFeatures]
        Tensor gradOutMat = gradOutput.reshape(batchSize, outFeatures);

        // Вход -> матрица [batchSize, inFeatures]
        Tensor inputMat = lastInput.reshape(batchSize, inFeatures);

        // Градиент по весам: gradW = input^T * gradOut
        Tensor inputT = inputMat.transpose(); // [inFeatures, batchSize]
        Tensor gradW = Tensor.matmul(inputT, gradOutMat); // [inFeatures, outFeatures]

        // Накапливаем градиент весов
        Tensor weightGrad = weights.grad.reshape(inFeatures, outFeatures);
        for (int i = 0; i < weightGrad.size; i++) {
            weightGrad.data[i] += gradW.data[i];
        }

        // Градиент по bias: сумма по батчу
        float[] gradBias = bias.grad.data;
        for (int i = 0; i < batchSize; i++) {
            int off = i * outFeatures;
            for (int j = 0; j < outFeatures; j++) {
                gradBias[j] += gradOutMat.data[off + j];
            }
        }

        // Градиент по входу: gradIn = gradOut * W^T
        Tensor weightsMat = weights.data.reshape(inFeatures, outFeatures);
        Tensor weightsT = weightsMat.transpose(); // [outFeatures, inFeatures]
        Tensor gradInMat = Tensor.matmul(gradOutMat, weightsT); // [batchSize, inFeatures]

        // Восстанавливаем исходную форму входа
        return gradInMat.reshape(lastInput.shape);
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(weights, bias);
    }
}
