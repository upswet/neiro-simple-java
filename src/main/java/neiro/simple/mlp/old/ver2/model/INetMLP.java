package neiro.simple.mlp.old.ver2.model;

import neiro.simple.mlp.old.ver2.WeightWrapper;
import neiro.simple.mlp.old.ver2.train.IWeightOptimaizer;

import java.util.function.Function;

/**Интерфейс полносвязанной нейросети*/
public interface INetMLP {
    default void init(Function<Double, WeightWrapper> createWeightWrapperFun){}

    /**Прямое распространение сигнала (вычисление)
     * @param inputs - вектор входных значений
     * @return -вектор выходных значений*/
    public double[] forward(double[] inputs);

    /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения
     * @param target - вектор целевых значений выходных нейронов
     * @param optimaizer - используемый при обучении оптимизатор весовых коэффициентов
     * @param isBatchFlg - если истина то обучение в пакетном режиме. Иначе нет
     * @param endBatchFlg - Только для пакетного режима обучения. Если истина, то данный батч закончился и необходимо скорректировать веса
     * @param batchCurrentSize - Только для пакетного режима обучения. Текущий размер пачки*/
    public void backward(double[] target, IWeightOptimaizer optimaizer, boolean isBatchFlg, boolean endBatchFlg, int batchCurrentSize);
}
