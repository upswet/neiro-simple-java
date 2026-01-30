package neiro.simple.obj;

import java.io.Serializable;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**Выходной слой*/
public class LayerOutput extends Layer {
    UnaryOperator<Double> activation; //функция активации
    Function<Neiron, Double> activationDer; //производная функции активации


    /**Конструктор выходного слоя
     * @param nCount - количество нейронов в слое
     * @param prevoisLayer - предыдущий слой
     * @param activation - фун активации
     * @param activationDer - производная фун активации
     * @param initWeightFun - функция инициализации весов
     * @return - выходной слой*/
    public LayerOutput(int nCount, Layer prevoisLayer, UnaryOperator<Double> activation, Function<Neiron, Double> activationDer, Supplier<Double> initWeightFun){
        this.activation =activation;
        this.activationDer =activationDer;

        for(int i =0; i<nCount; i++) {
            Neiron neiron = Neiron.createOutputNeiron();
            this.neirons.add(neiron);

            for(Neiron prevNeiron :  prevoisLayer.neirons)
                Link.createLink(prevNeiron, neiron, initWeightFun);
        }
    }

    /**Прямой проход для нейронов слоя (вычисление)
     * @return - вектор выходных данных*/
    public double[] forward(){
        double[] outputs = new double[neirons.size()];

        for(int i=0; i<neirons.size(); i++) {
            Neiron n = neirons.get(i);

            n.iValue=n.b;
            for (Link link : n.iLinks)
                n.iValue  += link.iNeiron.oValue * link.w;
            n.oValue=this.activation.apply(n.iValue);
            outputs[i]=n.oValue;
        }

        return outputs;
    }

    /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения
     * @param targets - вектор целевых значений выходных нейронов
     * @param lr - коэффициент обучения
     * */
    public void backward(double[] targets, double lr){
        assert targets.length!=neirons.size() : "Несовпадение размерности";

        //Вычислим дельту ошибки нейронов выходного слоя
        for(int i=0; i<targets.length; i++){
            Neiron neiron = neirons.get(i);

            //дельта ошибки на выходе нейрона = производная функции потерь * производная функции активации
            calcDelta(neiron, targets[i]);

            //вычисляем изменение смещения
            neiron.b -=neiron.delta*lr;

            //корректируем веса входящих связей для нейрона
            for (Link inLink : neiron.iLinks) {
                double grad = inLink.iNeiron.oValue * neiron.delta;
                inLink.w -=lr * grad;
            }
        }
    }

    /**Вычисление дельты выходного нейрона*/
    protected void calcDelta(Neiron neiron, double targets) {
        //дельта ошибки на выходе нейрона = производная функции потерь * производная функции активации
        //для  MSE  loss = (target - output)^2 производная по output: 2*(output - target) (но обычно берут (output - target))
        neiron.delta=(neiron.oValue - targets) * activationDer.apply(neiron);
    }

    /**Выходной слой - сигмоида*/
    public static class LayerOutputSigmoid extends LayerOutput{
        public LayerOutputSigmoid(Layer prevoisLayer, int nCount){
            super(nCount, prevoisLayer, Net.SIGMOID, Net.SIGMOID_DERIVATIVE, Net.SIGMOID_INIT(prevoisLayer.neirons.size()));
        }
    }

    /**Выходной слой - релу*/
    public static class LayerOutputRelu extends LayerOutput{
        public LayerOutputRelu(Layer prevoisLayer, int nCount){
            super(nCount, prevoisLayer, Net.RELU, Net.RELU_DERIVATIVE, Net.RELU_INIT(prevoisLayer.neirons.size()));
        }
    }
}

