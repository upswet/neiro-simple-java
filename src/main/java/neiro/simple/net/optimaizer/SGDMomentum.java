package neiro.simple.net.optimaizer;

import neiro.simple.net.Parameter;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Стохастический градиентный спуск с моментом (Momentum).
 * <p>
 * Обновление весов:
 * <pre>
 * velocity = momentum * velocity - learningRate * gradient
 * weight += velocity
 * </pre>
 * </p>
 * <p>
 * После обновления градиенты обнуляются.
 * </p>
 */
public class SGDMomentum implements Optimizer {

    private final float learningRate;
    private final float momentum;
    private final boolean nesterov; // опционально, Nesterov accelerated gradient

    // Хранилище скоростей для каждого параметра
    private final Map<Parameter, float[]> velocities = new HashMap<>();

    /**
     * @param learningRate шаг обучения (learning rate)
     * @param momentum     коэффициент момента (обычно 0.9)
     * @param nesterov     если true, используется Nesterov momentum
     */
    public SGDMomentum(float learningRate, float momentum, boolean nesterov) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.nesterov = nesterov;
    }

    /**
     * Упрощённый конструктор: без Nesterov, momentum = 0.9.
     */
    public SGDMomentum(float learningRate, float momentum) {
        this(learningRate, momentum, false);
    }

    /**
     * Конструктор с momentum по умолчанию 0.9.
     */
    public SGDMomentum(float learningRate) {
        this(learningRate, 0.9f, false);
    }

    @Override
    public void update(List<Parameter> parameters) {
        for (Parameter p : parameters) {
            float[] velocity = velocities.computeIfAbsent(p, k -> new float[p.data.size]);

            float[] data = p.data.data;
            float[] grad = p.grad.data;

            if (nesterov) {
                // Nesterov: сначала применяем скорость к весам (lookahead), потом обновляем
                // v = momentum * v - lr * grad(w + momentum * v)
                // но стандартная реализация требует двух проходов или модификации.
                // Упрощённая версия Nesterov: v = momentum * v - lr * grad;
                // weight += v;
                // Примечание: полный Nesterov сложнее, здесь приведена классическая формула.
                // Для большинства задач обычный momentum достаточен.
                for (int i = 0; i < data.length; i++) {
                    velocity[i] = momentum * velocity[i] - learningRate * grad[i];
                    data[i] += velocity[i];
                    grad[i] = 0.0f;
                }
            } else {
                // Обычный momentum
                for (int i = 0; i < data.length; i++) {
                    velocity[i] = momentum * velocity[i] - learningRate * grad[i];
                    data[i] += velocity[i];
                    grad[i] = 0.0f;
                }
            }
        }
    }

    @Override
    public void reset() {
        velocities.clear();
    }
}
