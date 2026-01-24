package neiro.simple.obj;

import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**Абстрактный класс слоя*/
public abstract class Layer implements Serializable{
    /**Нейрон*/
    @FieldDefaults(level = AccessLevel.PUBLIC)
    public static class Neiron implements Serializable{
        double b = 0F; //смещение
        double iValue=0F;//входное значение  (взвешенная сумма переданных сигналов от нейронов предыдущего слоя) + смещеие
        double oValue;//выходное значение - то что передаётся от этого нейрона нейрону следующего слоя.
        double delta; //дельта ошибки
        List<Link> iLinks = new ArrayList<>();
        List<Link> oLinks = new ArrayList<>();
        
        /**Создать нейрон входного слоя*/
        public static Neiron createInputNeiron(){Neiron n = new Neiron();n.iLinks=null;return n;}
        /**Создать нейрон промежуточного слоя*/
        public static Neiron createMediumNeiron(){return new Neiron();}
        /**Создать нейрон выходного слоя*/
        public static Neiron createOutputNeiron(){Neiron n = new Neiron();n.oLinks=null;return n;}

        public void print(int layerNumber, int neironNumber){
            System.out.println("\t\tneiron "+layerNumber+"_"+neironNumber+" ("+"b="+String.format("%.4f", b)+", delta="+String.format("%.4f",delta)+", iValue="+String.format("%.4f",iValue)+", oValue="+String.format("%.4f",oValue)+")");
            if(oLinks==null) return;
            for(int i=0; i<oLinks.size(); i++)
                oLinks.get(i).print(layerNumber, neironNumber, i);
        }
    }
    /**Связь между нейронами*/
    public static class Link implements Serializable{
        double w = 0F; //вес связи
        Neiron iNeiron;
        Neiron oNeiron;

        /**Создать связь между нейронами
         * @param iNeiron - входной нейрон
         * @param oNeiron - выходной нейрон
         * @param initWeightFun - функция инициализации весов
         * @return - созданная связь*/
        public static Link createLink(Neiron iNeiron, Neiron oNeiron, Supplier<Double> initWeightFun){
            Link link = new Link();
            link.w = initWeightFun.get();
            link.iNeiron = iNeiron;
            link.oNeiron = oNeiron;

            iNeiron.oLinks.add(link);
            oNeiron.iLinks.add(link);
            return link;
        }

        public void print(int layerNumber, int neironNumber, int linkNumber){
            System.out.println("\t\t\tlink "+layerNumber+"_"+neironNumber+" => "+(layerNumber+1)+"_"+linkNumber+" : "+String.format("%.4f", w));
        }
    }

    final List<Neiron> neirons = new ArrayList<>();

    /**Вывести на печать*/
    public void print(int layerNumber){
        System.out.println("\tLayer number "+layerNumber);
        for(int i=0; i<neirons.size(); i++)
            neirons.get(i).print(layerNumber, i);

    }
}
