package neiro.simple.mlp.stabile.model;

import neiro.simple.mlp.stabile.WeightWrapper;
import neiro.simple.mlp.stabile.train.IWeightOptimaizer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**Модель нейросети класса MLP (Полносвязнная нейроная сеть). Каждый нейрон реализован как объект. Без матричных вычислений*/
public class Net implements Serializable {
    List<Layer> layers = new ArrayList<>();

    /**Конструктор нейросети
     * @param layerCreateFuns - список функций создания слоя с получением в качестве аргумента предыдущего слоя
     * @return - нейросеть*/
    public Net(List<Function<Layer, Layer>> layerCreateFuns){
        layers.add(layerCreateFuns.getFirst().apply(null));
        for(int i=1; i<layerCreateFuns.size(); i++)
            layers.add(layerCreateFuns.get(i).apply(layers.getLast()));

        if (!Layer.input.class.isAssignableFrom(layers.getFirst().getClass()))
            throw new RuntimeException("Первый слой должен быть входным!");
        for(int i=1; i<layerCreateFuns.size()-1; i++)
            if (!Layer.medium.class.isAssignableFrom(layers.get(i).getClass()))
                throw new RuntimeException("Между входным и выходным-и слоями должны быть только промежуточные слои");
        if (!Layer.output.class.isAssignableFrom(layers.getLast().getClass()))
            throw new RuntimeException("Последний слой должен быть выходным!");
    }

    /**Инициализация веса
     * @param createWeightWrapperFun - функция создания WeightWrapper*/
    public void init(Function<Double, WeightWrapper> createWeightWrapperFun){
        for(Layer layer : layers)
            layer.init(createWeightWrapperFun);
    }

    /**Прямое распространение сигнала (вычисление)
     * @param inputs - вектор входных значений
     * @return -вектор выходных значений*/
    public double[] forward(double[] inputs){
        ((Layer.input)layers.getFirst()).forward(inputs);

        for (int i=1; i<layers.size()-1; i++)
            ((Layer.medium)layers.get(i)).forward();

        return ((Layer.output)layers.getLast()).forward();
    }

    /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения
     * @param target - вектор целевых значений выходных нейронов
     * @param optimaizer - используемый при обучении оптимизатор весовых коэффициентов
     * @param isBatchFlg - если истина то обучение в пакетном режиме. Иначе нет
     * @param endBatchFlg - Только для пакетного режима обучения. Если истина, то данный батч закончился и необходимо скорректировать веса
     * @param batchCurrentSize - Только для пакетного режима обучения. Текущий размер пачки*/
    public void backward(double[] target, IWeightOptimaizer optimaizer, boolean isBatchFlg, boolean endBatchFlg, int batchCurrentSize){
        /*
        Ошибка вычисляется на выходном слое.
        Затем она распространяется назад: от выходного слоя к входному.
        Каждый нейрон (кроме входных) имеет дельту (δ), которая показывает, насколько он виноват в ошибке.

        Layer.output.backward:
            - вычисляет дельту для каждого нейрона выходного слоя на основе разницы между целевыми и полученными данными
            - обновляет веса входящих связей (iLinks) этих нейронов.
        Layer.medium.backward:
            - вычисляет дельту для каждого нейрона скрытого слоя (используя дельты следующего слоя и веса исходящих связей).
            - обновляет веса входящих связей (iLinks) этих нейронов.
        Layer.input.backward:
            - Входной слой не имеет входящих связей, а его исходящие связи (oLinks) уже обновлены на предыдущем шаге. Ничего не делаем
        */

        ((Layer.output)layers.getLast()).backward(target, optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize);

        for (int i=layers.size()-2; i>0; i--)
            ((Layer.medium)layers.get(i)).backward(optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize);

        // Ничего не делаем так как исходящие из входного слоя связи уже обновлены первым промежуточным слоем, а дельты вычислять не надо так как входящих связей нет
        // ((Layer.input)layers.getFirst()).backward();
    }
}
