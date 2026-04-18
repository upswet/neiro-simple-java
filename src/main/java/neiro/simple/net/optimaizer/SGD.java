package neiro.simple.net.optimaizer;

import neiro.simple.net.Parameter;

import java.util.List;

/**
 * Простой стохастический градиентный спуск: w = w - lr * grad.
 */
public class SGD implements Optimizer {
    private final float learningRate;

    public SGD(float learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void update(List<Parameter> parameters) {
        for (Parameter p : parameters) {
            for (int i = 0; i < p.data.size; i++) {
                p.data.data[i] -= learningRate * p.grad.data[i];
                p.grad.data[i] = 0.0f; // сбрасываем градиент
            }
        }
    }

    @Override
    public void reset() {
        // нет состояния
    }
}