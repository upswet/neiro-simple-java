package neiro.simple.net.layer.fun;

import neiro.simple.net.Tensor;
import neiro.simple.net.layer.Layer;

/**
 * Сигмоида: f(x) = 1 / (1 + exp(-x)).
 */
public class SigmoidLayer implements Layer {
    private Tensor lastOutput; // y = sigmoid(x)

    @Override
    public Tensor forward(Tensor input) {
        lastOutput = input.map(x -> {
            if (x >= 0) {
                return 1.0f / (1.0f + (float) Math.exp(-x));
            } else {
                float expX = (float) Math.exp(x);
                return expX / (1.0f + expX);
            }
        });
        return lastOutput;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // производная: y * (1 - y)
        Tensor gradInput = gradOutput.copy();
        for (int i = 0; i < gradInput.size; i++) {
            float y = lastOutput.data[i];
            gradInput.data[i] *= y * (1.0f - y);
        }
        return gradInput;
    }
}
