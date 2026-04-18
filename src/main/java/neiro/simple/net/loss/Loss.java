package neiro.simple.net.loss;


import neiro.simple.net.Tensor;

/**
 * Интерфейс функции потерь.
 */
public interface Loss {

    /**
     * Вычислить значение функции потерь.
     *
     * @param predicted предсказания модели
     * @param target    истинные значения
     * @return скалярное значение (среднее по батчу)
     */
    float compute(Tensor predicted, Tensor target);

    /**
     * Вычислить градиент функции потерь по предсказаниям.
     *
     * @param predicted предсказания модели
     * @param target    истинные значения
     * @return тензор градиентов той же формы, что и predicted
     */
    Tensor gradient(Tensor predicted, Tensor target);
}