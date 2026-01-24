package neiro.simple.obj;

import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**Промежуточный слой*/
public class LayerMedium extends Layer {
    UnaryOperator<Double> activation; //функция активации
    Function<Neiron, Double> activationDer; //производная функции активации

    /**Конструктор промежуточного слоя
     * @param nCount - количество нейронов в слое
     * @param prevoisLayer - предыдущий слой
     * @param activation - фун активации
     * @param activationDer - производная фун активации
     * @param initWeightFun - функция инициализации весов
     * @return - промежуточный слой*/
    public LayerMedium (int nCount, Layer prevoisLayer, UnaryOperator<Double> activation, Function<Neiron, Double> activationDer, Supplier<Double> initWeightFun){
        this.activation =activation;
        this.activationDer =activationDer;

        for(int i =0; i<nCount; i++) {
            Neiron neiron = Neiron.createMediumNeiron();
            this.neirons.add(neiron);

            for(Neiron prevNeiron :  prevoisLayer.neirons)
                Link.createLink(prevNeiron, neiron, initWeightFun);
        }

    }


    /**Прямой проход для нейронов слоя (вычисление)*/
    public void forward(){
        for(Neiron n : neirons) {
            n.iValue=n.b;
            for (Link link : n.iLinks)
                link.oNeiron.iValue += link.iNeiron.oValue * link.w;
            n.oValue=this.activation.apply(n.iValue);
        }
    }

    /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения
     * @param lr - коэффициент обучения*/
    public void backward(double lr){
        for (Neiron neiron : neirons) {
            //Вычислим дельту ошибки чтобы в предыдущем слоя можно было обновить веса исходящих из него (то есть входящих для нейронов данного слоя) связей
            //для всех нейронов данного слоя обновим веса исходящих связей на основе переданного нейроном значения, веса связи и дельты ошибки нейрона-получателя данной связи

            double delta = 0F;
            for (Link outLink : neiron.oLinks)
                delta += outLink.oNeiron.delta * outLink.w;
            neiron.delta = activationDer.apply(neiron) * delta;

            neiron.b -= neiron.delta*lr;//вычисление изменения смещения текущего нейрона

            //корректируем веса входящих связей для нейрона
            for (Link inLink : neiron.iLinks) {
                double grad = inLink.iNeiron.oValue * neiron.delta;
                inLink.w -= lr * grad;
            }
        }
    }

    /**Промежуточный слой - сигмоида*/
    public static class LayerMediumSigmoid extends LayerMedium{
        public LayerMediumSigmoid(Layer prevoisLayer, int nCount){
            super(nCount, prevoisLayer, Net.SIGMOID, Net.SIGMOID_DERIVATIVE, Net.SIGMOID_INIT(prevoisLayer.neirons.size()));
        }
    }
    /**Промежуточный слой - релу*/
    public static class LayerMediumRelu extends LayerMedium{
        public LayerMediumRelu(Layer prevoisLayer, int nCount){
            super(nCount, prevoisLayer, Net.RELU, Net.RELU_DERIVATIVE, Net.RELU_INIT(prevoisLayer.neirons.size()));
        }
    }
}
