package neiro.simple.mlp.old.ver2.task;

import neiro.simple.mlp.old.ver2.dto.Example;
import neiro.simple.mlp.old.ver2.dto.Examples;

import java.util.List;
import java.util.stream.IntStream;

/**Интерфейс описывающий "задачу"*/
public interface ITask {
    /**Подготовить данные для обучения и тестирования нейросети
     * @param path - путь к дирректории с "сырыми" данными
     * @return - подготовленные данные*/
    Examples getData(String path);


    /**Преобразуем списки входных и целевых векторов в массив примеров Example[]*/
    default Example[] convertToExampleArray(List<double[]> inputs, List<double[]> targets){
        return IntStream.range(0, inputs.size())
                .mapToObj(i -> new Example(inputs.get(i), targets.get(i)))
                .toArray(Example[]::new);
    }
}
