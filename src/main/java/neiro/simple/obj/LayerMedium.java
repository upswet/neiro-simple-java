package neiro.simple.obj;

import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**Промежуточный слой*/
public class LayerMedium extends Layer {
    UnaryOperator<Double> activation; //функция активации
    Function<Neiron, Double> activationDer; //производная функции активации

    double dropoutRate = 0.0; // процент дропаута (0.0 - нет дропаута, 0.5 - 50%)

    /**Конструктор промежуточного слоя
     * @param nCount - количество нейронов в слое
     * @param prevoisLayer - предыдущий слой
     * @param activation - фун активации
     * @param activationDer - производная фун активации
     * @param initWeightFun - функция инициализации весов
     * @param dropoutRate - процент дропаута (от 0.0 до 0.99)
     * @return - промежуточный слой*/
    public LayerMedium (int nCount, Layer prevoisLayer, UnaryOperator<Double> activation, Function<Neiron, Double> activationDer, Supplier<Double> initWeightFun, double dropoutRate){
        this.activation =activation;
        this.activationDer =activationDer;
        this.dropoutRate = dropoutRate;

        for(int i =0; i<nCount; i++) {
            Neiron neiron = Neiron.createMediumNeiron();
            this.neirons.add(neiron);

            for(Neiron prevNeiron :  prevoisLayer.neirons)
                Link.createLink(prevNeiron, neiron, initWeightFun);
        }

    }


    /**Прямой проход для нейронов слоя (вычисление)
     * @param trainingMode - если истина, то режим обучения, иначе режим работы*/
    public void forward(boolean trainingMode){
        for(Neiron n : neirons) {
            if (trainingMode && dropoutRate > 0.0) {
                if (Math.random() < dropoutRate) {
                    n.dropoutMask = 0.0; // нейрон отключен
                    n.iValue = 0.0;
                    n.oValue = 0.0;
                    continue;
                }
                else
                    n.dropoutMask = 1.0; // нейрон активен
            }else
                n.dropoutMask = 1.0; // вне режима тренировки нейрон всегда активен


            n.iValue=n.b;
            for (Link link : n.iLinks)
                n.iValue  += link.iNeiron.oValue * link.w;
            n.oValue=this.activation.apply(n.iValue);

            // Для inverted dropout: масштабируем только при обучении
            if (trainingMode && dropoutRate > 0.0 && n.dropoutMask == 1.0) {
                n.oValue *= 1.0 / (1.0 - dropoutRate); // inverted dropout
            }
        }
    }

    /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения
     * @param lr - коэффициент обучения
     * @param trainingMode - если истина, то режим обучения, иначе режим работы*/
    public void backward(double lr, boolean trainingMode){
        for (Neiron neiron : neirons) {
            // Учитываем маску дропаута при вычислении градиента
            if (neiron.dropoutMask == 0.0) {
                neiron.delta = 0.0; // отключенные нейроны не участвуют в обучении
                continue;
            }

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
        public LayerMediumSigmoid(Layer prevoisLayer, int nCount, double dropoutRate){
            super(nCount, prevoisLayer, Net.SIGMOID, Net.SIGMOID_DERIVATIVE, Net.SIGMOID_INIT_XAVIER(prevoisLayer.neirons.size(), nCount), dropoutRate);
        }
    }
    /**Промежуточный слой - тангес*/
    public static class LayerMediumTanh extends LayerMedium{
        public LayerMediumTanh(Layer prevoisLayer, int nCount, double dropoutRate){
            super(nCount, prevoisLayer, Net.TANH, Net.TANH_DERIVATIVE, Net.TANH_INIT_XAVIER(prevoisLayer.neirons.size(), nCount), dropoutRate);
        }
    }
    /**Промежуточный слой - релу*/
    public static class LayerMediumRelu extends LayerMedium{
        public LayerMediumRelu(Layer prevoisLayer, int nCount, double dropoutRate){
            super(nCount, prevoisLayer, Net.RELU, Net.RELU_DERIVATIVE, Net.RELU_INIT_HE(prevoisLayer.neirons.size()), dropoutRate);
        }
    }
}
