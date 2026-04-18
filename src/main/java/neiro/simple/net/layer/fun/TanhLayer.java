package neiro.simple.net.layer.fun;

import neiro.simple.net.Tensor;
import neiro.simple.net.layer.Layer;

/**
 * Гиперболический тангенс: f(x) = tanh(x).
 */
public class TanhLayer implements Layer {
    private Tensor lastOutput;

    @Override
    public Tensor forward(Tensor input) {
        lastOutput = input.map(x -> (float) Math.tanh(x));
        return lastOutput;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        Tensor gradInput = gradOutput.copy();
        for (int i = 0; i < gradInput.size; i++) {
            float y = lastOutput.data[i];
            gradInput.data[i] *= (1.0f - y * y);
        }
        return gradInput;
    }
}
