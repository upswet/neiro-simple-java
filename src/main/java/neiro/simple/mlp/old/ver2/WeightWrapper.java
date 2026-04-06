package neiro.simple.mlp.old.ver2;

import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import neiro.simple.mlp.old.ver2.train.IWeightOptimaizer;

import java.io.Serializable;

/**Враппер для весового коэффициента*/
@FieldDefaults(level = AccessLevel.PUBLIC)
public abstract class WeightWrapper implements Serializable {
    double item; //собственно значение веса

    double accum=0.0;//Здесь будут накапливаться градиенты при пакетных методах обучения

    public WeightWrapper(double item) {this.item = item;}

    /**Обработка изменения веса не в пакетном режиме
     * @param grad - градиент изменения веса
     * @param optimaizer - используемый при обучении оптимизатор весовых коэффициентов*/
    protected void processNoBatch(double grad, IWeightOptimaizer optimaizer){
        optimaizer.update(this, grad);
    }

    /**Обработка изменения веса в пакетном режиме
     * @param grad - градиент изменения веса
     * @param optimaizer - используемый при обучении оптимизатор весовых коэффициентов
     * @param endBatchFlg - если истина, то данный батч закончился и необходимо скорректировать веса
     * @param batchCurrentSize - текущий размер пачки (если -1 то без пакетного режима)*/
    protected void processBatch(double grad, IWeightOptimaizer optimaizer, boolean endBatchFlg, int batchCurrentSize){
        accum+=grad;

        if (endBatchFlg) {
            optimaizer.update(this, accum / batchCurrentSize);
            accum = 0.0;
        }
    }

    /**Обработка изменения веса в пакетном режиме и без
     * @param grad - градиент изменения веса
     * @param optimaizer - используемый при обучении оптимизатор весовых коэффициентов
     * @param isBatchFlg - если истина то обучение в пакетном режиме. Иначе нет
     * @param endBatchFlg - если истина, то данный батч закончился и необходимо скорректировать веса
     * @param batchCurrentSize - текущий размер пачки (если -1 то без пакетного режима)*/
    public void process(double grad, IWeightOptimaizer optimaizer, boolean isBatchFlg, boolean endBatchFlg, int batchCurrentSize){
        if (isBatchFlg)
            processBatch(grad, optimaizer, endBatchFlg, batchCurrentSize);
        else
            processNoBatch(grad, optimaizer);
    }

    /**Враппер для весового коэффициента без оптимизации*/
    public static class classic extends WeightWrapper {
        public classic(double item) {super(item);}
    }


        /**Враппер для весового коэффициента для оптимизатора типа Momentum*/
    public static class momentum extends WeightWrapper {
        /**Параметры для оптимизатора*/
        public double m = 0.0;  // момент

        public momentum(double item) {super(item);}
    }

    /**Враппер для весового коэффициента для оптимизатора типа АДАМ*/
    public static class adam extends WeightWrapper {
        /**Параметры для оптимизатора*/
        public double m = 0.0;  // первый момент
        public double v = 0.0;  // второй момент

        public adam(double item) {super(item);}
    }
}
