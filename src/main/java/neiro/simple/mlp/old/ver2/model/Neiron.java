package neiro.simple.mlp.old.ver2.model;

import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import neiro.simple.mlp.old.ver2.WeightWrapper;
import neiro.simple.mlp.old.ver2.train.IWeightOptimaizer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**Модель нейрона*/
@FieldDefaults(level = AccessLevel.PUBLIC)
public class Neiron implements Serializable{
    WeightWrapper bias; //смещение

    double iValue=0F;//входное значение (взвешенная сумма переданных сигналов от нейронов предыдущего слоя) + смещение
    double oValue;//выходное значение - то что передаётся от этого нейрона к нейрону следующего слоя.
    //double oValueRaw;//сырое выходное значение (без изменений в связи с дроп-аутом) только для промежуточных слоёв
    double delta; //дельта ошибки
    List<Link> iLinks = new ArrayList<>(); //входящие связи нейрона
    List<Link> oLinks = new ArrayList<>(); //исходящие связи нейрона

    /**Инициализация веса
     * @param createWeightWrapperFun - функция создания WeightWrapper
     * @param initFun - функция инициализации веса*/
    public void init(Function<Double, WeightWrapper> createWeightWrapperFun, Supplier<Double> initFun){
        this.bias = createWeightWrapperFun.apply(initFun.get());

        for(Link link : iLinks)
            link.init(createWeightWrapperFun, initFun);
    }

    /**Процесс работы нейрона: получить сигнал от всех своих входных связей и вычислить выходное значение
     * @param activation - функция активации нейрона*/
    public void process(UnaryOperator<Double> activation){
        iValue = bias.item;
        for (Link link : iLinks)
            iValue += link.iNeiron.oValue * link.weight.item;
        oValue = activation.apply(iValue);
    }

    /**Корректируем веса входящих связей для нейрона на основе ранее вычисленной дельты нейрона
     * @param optimaizer - используемый при обучении оптимизатор весовых коэффициентов
     * @param isBatchFlg - если истина то обучение в пакетном режиме. Иначе нет
     * @param endBatchFlg - Только для пакетного режима обучения. Если истина, то данный батч закончился и необходимо скорректировать веса
     * @param batchCurrentSize - Только для пакетного режима обучения. Текущий размер пачки*/
    public void correctWeightInputLink(IWeightOptimaizer optimaizer, boolean isBatchFlg, boolean endBatchFlg, int batchCurrentSize){
        for (Link inLink : iLinks){
            double grad = inLink.iNeiron.oValue * delta;
            inLink.weight.process(grad, optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize);
        }
    }
}
