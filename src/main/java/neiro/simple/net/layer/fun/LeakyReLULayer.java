package neiro.simple.net.layer.fun;

import neiro.simple.net.Tensor;
import neiro.simple.net.layer.Layer;

/**
 * Leaky ReLU: f(x) = x если x > 0, иначе alpha * x.
 */
public class LeakyReLULayer implements Layer {
    private final float alpha;
    private Tensor lastInput;

    public LeakyReLULayer(float alpha) {
        this.alpha = alpha;
    }

    public LeakyReLULayer() {
        this(0.01f);
    }

    @Override
    public Tensor forward(Tensor input) {
        lastInput = input.copy();
        return input.map(x -> x > 0 ? x : alpha * x);
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        Tensor gradInput = gradOutput.copy();
        for (int i = 0; i < gradInput.size; i++) {
            float x = lastInput.data[i];
            gradInput.data[i] *= (x > 0 ? 1.0f : alpha);
        }
        return gradInput;
    }
}