package neiro.simple.mlp.stabile.model;

import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import neiro.simple.mlp.stabile.WeightWrapper;

import java.io.Serializable;
import java.util.function.Function;
import java.util.function.Supplier;

/**Модель связи между нейронами*/
@FieldDefaults(level = AccessLevel.PUBLIC)
public class Link implements Serializable {
    WeightWrapper weight;

    Neiron iNeiron; //нейрон откуда выходит связь
    Neiron oNeiron; //и нейрон куда входит связь

    /**Конструктор связи
     * @param iNeiron - нейрон откуда выходит связь
     * @param oNeiron - нейрон куда вхдоит связь*/
    public Link(Neiron iNeiron, Neiron oNeiron) {
        this.iNeiron = iNeiron;
        this.oNeiron = oNeiron;


        iNeiron.oLinks.add(this);
        oNeiron.iLinks.add(this);
    }

    /**Инициализация веса
     * @param createWeightWrapperFun - функция создания WeightWrapper
     * @param initFun - функция инициализации веса*/
    public void init(Function<Double, WeightWrapper> createWeightWrapperFun, Supplier<Double> initFun){
        this.weight = createWeightWrapperFun.apply(initFun.get());
    }
}
