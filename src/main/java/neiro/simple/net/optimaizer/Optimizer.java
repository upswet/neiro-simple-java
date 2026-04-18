package neiro.simple.net.optimaizer;

import neiro.simple.net.Parameter;

import java.util.List;

/**
 * Интерфейс оптимизатора, обновляющего параметры модели на основе накопленных градиентов.
 */
public interface Optimizer {

    /**
     * Обновить все переданные параметры.
     * После вызова градиенты обычно обнуляются.
     *
     * @param parameters список параметров для обновления
     */
    void update(List<Parameter> parameters);

    /**
     * Сбросить внутреннее состояние (например, моменты в Adam) для нового цикла обучения.
     */
    void reset();
}
