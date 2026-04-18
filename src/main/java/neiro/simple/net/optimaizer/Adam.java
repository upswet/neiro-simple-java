package neiro.simple.net.optimaizer;

import neiro.simple.net.Parameter;

import java.util.*;

/**
 * Оптимизатор Adam (Adaptive Moment Estimation).
 * <p>
 * Поддерживает экспоненциальные скользящие средние градиента (m) и квадрата градиента (v).
 */
public class Adam implements Optimizer {
    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private int t; // счётчик шагов (общий для всех параметров)

    private final Map<Parameter, float[]> m = new HashMap<>();
    private final Map<Parameter, float[]> v = new HashMap<>();

    public Adam(float learningRate, float beta1, float beta2, float epsilon) {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.t = 0;
    }

    @Override
    public void update(List<Parameter> parameters) {
        t++;
        float beta1t = (float) Math.pow(beta1, t);
        float beta2t = (float) Math.pow(beta2, t);
        float alpha = learningRate * (float) Math.sqrt(1 - beta2t) / (1 - beta1t);

        for (Parameter p : parameters) {
            float[] mt = m.computeIfAbsent(p, k -> new float[p.data.size]);
            float[] vt = v.computeIfAbsent(p, k -> new float[p.data.size]);
            float[] data = p.data.data;
            float[] grad = p.grad.data;

            for (int i = 0; i < data.length; i++) {
                float g = grad[i];
                mt[i] = beta1 * mt[i] + (1 - beta1) * g;
                vt[i] = beta2 * vt[i] + (1 - beta2) * g * g;
                float mHat = mt[i] / (1 - beta1t);
                float vHat = vt[i] / (1 - beta2t);
                data[i] -= alpha * mHat / ((float) Math.sqrt(vHat) + epsilon);
                grad[i] = 0.0f; // обнуляем градиент
            }
        }
    }

    @Override
    public void reset() {
        t = 0;
        m.clear();
        v.clear();
    }
}
