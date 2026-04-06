package neiro.simple.mlp.old.ver3Fast;

import neiro.simple.mlp.old.ver2.WeightWrapper;
import neiro.simple.mlp.old.ver2.model.INetMLP;
import neiro.simple.mlp.old.ver2.train.IWeightOptimaizer;

import java.io.Serializable;
import java.util.List;
import java.util.function.Function;

/**Ускоренная нейросеть*/
public class NetFast implements INetMLP, Serializable {
    LayerFast[] layers;

    public NetFast(List<Function<LayerFast, LayerFast>> layerCreateFuns){
        layers = new LayerFast[layerCreateFuns.size()];

        layers[0]=layerCreateFuns.get(0).apply(null);
        for(int i=1; i<layerCreateFuns.size(); i++)
            layers[i]=layerCreateFuns.get(i).apply(layers[i-1]);

        if (!LayerFast.input.class.isAssignableFrom(layers[0].getClass()))
            throw new RuntimeException("Первый слой должен быть входным!");
        for(int i=1; i<layerCreateFuns.size()-1; i++)
            if (!LayerFast.medium.class.isAssignableFrom(layers[i].getClass()))
                throw new RuntimeException("Между входным и выходным-и слоями должны быть только промежуточные слои");
        if (!LayerFast.output.class.isAssignableFrom(layers[layers.length-1].getClass()))
            throw new RuntimeException("Последний слой должен быть выходным!");
    }

    /**Инициализация веса
     * @param createWeightWrapperFun - функция создания WeightWrapper*/
    public void init(Function<Double, WeightWrapper> createWeightWrapperFun){
        for(int i=1; i<layers.length; i++)
            ((LayerFast.noInput)layers[i]).init(createWeightWrapperFun);
    }

    /**Прямое распространение сигнала (вычисление)
     * @param inputs - вектор входных значений
     * @return -вектор выходных значений*/
    public double[] forward(double[] inputs){
        ((LayerFast.input)layers[0]).forward(inputs);

        for (int i=1; i<layers.length-1; i++)
            ((LayerFast.medium)layers[i]).forward();

        return ((LayerFast.output)layers[layers.length-1]).forward();
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

        ((LayerFast.output)layers[layers.length-1]).backward(target, optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize);

        for (int i=layers.length-2; i>0; i--)
            ((LayerFast.medium)layers[i]).backward(null,optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize);

        // Ничего не делаем так как исходящие из входного слоя связи уже обновлены первым промежуточным слоем, а дельты вычислять не надо так как входящих связей нет
        // ((LayerFast.input)layers.getFirst()).backward();
    }
}
