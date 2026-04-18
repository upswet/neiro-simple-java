package neiro.simple.net.layer.fun;

import neiro.simple.net.Tensor;
import neiro.simple.net.layer.Layer;

/**
 * ReLU (Rectified Linear Unit): f(x) = max(0, x).
 */
public class ReLULayer implements Layer {
    private Tensor lastInput; // для backward (нужно знать, где x <= 0)

    @Override
    public Tensor forward(Tensor input) {
        lastInput = input.copy(); // сохраняем копию, т.к. backward требует исходные значения
        return input.map(x -> x > 0 ? x : 0.0f);
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // копируем, т.к. будем модифицировать
        Tensor gradInput = gradOutput.copy();
        for (int i = 0; i < gradInput.size; i++) {
            if (lastInput.data[i] <= 0) {
                gradInput.data[i] = 0.0f;
            }
        }
        return gradInput;
    }
}
