package neiro.simple.net.metric;

import neiro.simple.net.Tensor;

/**
 * Интерфейс метрики качества.
 */
public interface Metric {
    void update(Tensor predicted, Object target);
    void reset();
    float getValue();
    String name();
}