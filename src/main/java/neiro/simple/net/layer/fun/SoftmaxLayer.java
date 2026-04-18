package neiro.simple.net.layer.fun;

import neiro.simple.net.Tensor;
import neiro.simple.net.layer.Layer;

/**
 * Softmax активация. Применяется вдоль последнего измерения.
 * Используется в задачах классификации с взаимоисключающими классами.
 */
public class SoftmaxLayer implements Layer {
    private Tensor lastOutput;
    private int[] lastShape;

    @Override
    public Tensor forward(Tensor input) {
        lastShape = input.shape.clone();
        int rank = input.rank;
        int batchSize = 1;
        for (int i = 0; i < rank - 1; i++) batchSize *= input.shape[i];
        int numClasses = input.shape[rank - 1];

        Tensor input2D = input.reshape(batchSize, numClasses);
        Tensor output2D = new Tensor(batchSize, numClasses);

        for (int i = 0; i < batchSize; i++) {
            // Находим максимум для численной стабильности
            int rowOff = i * numClasses;
            float max = input2D.data[rowOff];
            for (int j = 1; j < numClasses; j++) {
                if (input2D.data[rowOff + j] > max) max = input2D.data[rowOff + j];
            }
            float sum = 0.0f;
            for (int j = 0; j < numClasses; j++) {
                float expVal = (float) Math.exp(input2D.data[rowOff + j] - max);
                output2D.data[rowOff + j] = expVal;
                sum += expVal;
            }
            for (int j = 0; j < numClasses; j++) {
                output2D.data[rowOff + j] /= sum;
            }
        }

        lastOutput = output2D.reshape(lastShape);
        return lastOutput;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // Общий случай (не в паре с кросс-энтропией)
        int rank = lastShape.length;
        int batchSize = 1;
        for (int i = 0; i < rank - 1; i++) batchSize *= lastShape[i];
        int numClasses = lastShape[rank - 1];

        Tensor gradOut2D = gradOutput.reshape(batchSize, numClasses);
        Tensor prob2D = lastOutput.reshape(batchSize, numClasses);
        Tensor gradIn2D = new Tensor(batchSize, numClasses);

        for (int b = 0; b < batchSize; b++) {
            int off = b * numClasses;
            // sum = Σ gradOut_i * prob_i
            float sumGradProb = 0.0f;
            for (int i = 0; i < numClasses; i++) {
                sumGradProb += gradOut2D.data[off + i] * prob2D.data[off + i];
            }
            for (int i = 0; i < numClasses; i++) {
                float p = prob2D.data[off + i];
                gradIn2D.data[off + i] = p * (gradOut2D.data[off + i] - sumGradProb);
            }
        }
        return gradIn2D.reshape(lastShape);
    }
}
