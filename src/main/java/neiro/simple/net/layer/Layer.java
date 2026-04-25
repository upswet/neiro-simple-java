package neiro.simple.net.layer;

import neiro.simple.net.Parameter;
import neiro.simple.net.Tensor;

import java.util.Collections;
import java.util.List;

/**
 * Базовый интерфейс слоя нейронной сети.
 * Оперирует тензорами {@link Tensor}, поддерживает прямой и обратный проход.
 */
public interface Layer {

    /**
     * Прямой проход слоя.
     *
     * @param input входной тензор
     * @return выходной тензор
     */
    Tensor forward(Tensor input);

    /**
     * Обратный проход слоя.
     *
     * @param gradOutput градиент по выходу слоя (той же формы, что и результат forward)
     * @return градиент по входу слоя
     */
    Tensor backward(Tensor gradOutput);

    /**
     * Список обучаемых параметров слоя (может быть пустым).
     */
    default List<Parameter> parameters() {
        return Collections.emptyList();
    }

    /**
     * Инициализация параметров слоя с учётом соседних слоёв.
     *
     * @param prev предыдущий слой (может быть null)
     * @param next следующий слой (может быть null)
     */
    default void initializeParameters(Layer prev, Layer next){};

    default void setTraining(boolean training) {}
}
