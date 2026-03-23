package neiro.simple.mlp.stabile.model;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.experimental.FieldDefaults;
import neiro.simple.mlp.stabile.WeightWrapper;
import neiro.simple.mlp.stabile.train.IWeightOptimaizer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**Модель слоя*/
@RequiredArgsConstructor
public abstract class Layer implements Serializable {
    final public List<Neiron> neirons = new ArrayList<>();
    final public Supplier<Double> initFun;

    /**Создать слой (создать нейроны данного слоя и связи между нейронами предыдущего слоя и нейронами данного слоя, если предыдущий слой был)
     * @param nCount - количество нейронов в слое
     * @param prevoisLayer - предыдущий слой
     * */
    protected void createLayer(int nCount, Layer prevoisLayer) {
        for(int i = 0; i< nCount; i++) {
            Neiron neiron = new Neiron();
            this.neirons.add(neiron);

            if (prevoisLayer!=null)
                for(Neiron prevNeiron :  prevoisLayer.neirons)
                    new Link(prevNeiron, neiron);
        }
    }

    /**Инициализация веса
     * @param createWeightWrapperFun - функция создания WeightWrapper*/
    public void init(Function<Double, WeightWrapper> createWeightWrapperFun){
        if (initFun == null) return;

        for(Neiron neiron : neirons)
            neiron.init(createWeightWrapperFun, initFun);
    }

    /**Входной слой*/
    @FieldDefaults(level = AccessLevel.PUBLIC)
    public static class input extends Layer{
        /**Конструктор входного слоя
         * @param nCount - количество нейронов в слое*/
        public input(int nCount) {
            super(null);
            createLayer(nCount, null);
        }

        /**Прямой проход для нейронов слоя (вычисление)
         * @param input - вектор входных данных*/
        public void forward(double[] input){
            assert (input.length!=neirons.size()) : "Несовпадение размерности";

            for(int i=0; i<input.length; i++)
                neirons.get(i).oValue = input[i];
        }

        /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения*/
        public void backward(){
            // Ничего не делаем так как исходящие из входного слоя связи уже обновлены первым промежуточным слоем, а дельты вычислять не надо так как входящих связей нет
        }
    }

    /**Промежуточный слой*/
    @FieldDefaults(level = AccessLevel.PUBLIC)
    public static class medium extends Layer {
        UnaryOperator<Double> activation; //функция активации
        Function<Neiron, Double> derivative; //производная функции активации

        /**Конструктор промежуточного слоя
         * @param nCount - количество нейронов в слое
         * @param initFun - функция инициализации весов
         * @param prevoisLayer - предыдущий слой
         * @param activation - функция активации
         * @param derivative - производная функции активации
         * */
        public medium(int nCount, Supplier<Double> initFun, Layer prevoisLayer, UnaryOperator<Double> activation, Function<Neiron, Double> derivative) {
            super(initFun);
            this.activation = activation;
            this.derivative = derivative;

            createLayer(nCount, prevoisLayer);
        }

        /**Прямой проход для нейронов слоя (вычисление)*/
        public void forward(){
            for(Neiron neiron : neirons)
                neiron.process(activation);
        }

        /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения
         * @param optimaizer - используемый при обучении оптимизатор весовых коэффициентов
         * @param isBatchFlg - если истина то обучение в пакетном режиме. Иначе нет
         * @param endBatchFlg - Только для пакетного режима обучения. Если истина, то данный батч закончился и необходимо скорректировать веса
         * @param batchCurrentSize - Только для пакетного режима обучения. Текущий размер пачки*/
        public void backward(IWeightOptimaizer optimaizer, boolean isBatchFlg, boolean endBatchFlg, int batchCurrentSize){
            for (Neiron neiron : neirons){
                //Протащим дельту ошибки (вычислим для нейронов текущего слоя на основе уже известных дельт ошибок нейронов следующего слоя)
                neiron.calcDelta(derivative);

                //корректируем вес смещения для нейрона
                neiron.bias.process(neiron.delta, optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize); //производная по bias равна delta (так как bias влияет напрямую на iValue)

                //корректируем веса входящих связей для нейрона
                neiron.correctWeightInputLink(optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize);
            }
        }

        public static class sigmoid extends medium{ public sigmoid(int nCount, Layer prevoisLayer) {super(nCount, Fun.INIT_XAVIER(prevoisLayer.neirons.size(), nCount), prevoisLayer, Fun.SIGMOID, Fun.SIGMOID_DERIVATIVE);}}
        public static class tanh extends medium{ public tanh(int nCount, Layer prevoisLayer) {super(nCount, Fun.INIT_XAVIER(prevoisLayer.neirons.size(), nCount), prevoisLayer, Fun.TANH, Fun.TANH_DERIVATIVE);}}
        public static class relu extends medium{ public relu(int nCount, Layer prevoisLayer) {super(nCount, Fun.INIT_HE(prevoisLayer.neirons.size()), prevoisLayer, Fun.RELU, Fun.RELU_DERIVATIVE);}}
    }

    /**Выходной слой*/
    @FieldDefaults(level = AccessLevel.PUBLIC)
    public static class output extends Layer {
        UnaryOperator<Double> activation; //функция активации
        Function<Neiron, Double> derivative; //производная функции активации
        BiFunction<Neiron, Double, Double> derivativeLoss; //производная функции потерь

        /**
         * Конструктор выходного слоя
         * @param nCount                 - количество нейронов в слое
         * @param initFun - функция инициализации весов
         * @param prevoisLayer           - предыдущий слой
         * @param activation             - функция активации
         * @param derivative             - производная функции активации
         * @param derivativeLoss - производная функции потерь
         */
        public output(int nCount, Supplier<Double> initFun, Layer prevoisLayer, UnaryOperator<Double> activation, Function<Neiron, Double> derivative, BiFunction<Neiron, Double, Double> derivativeLoss) {
            super(initFun);
            this.activation = activation;
            this.derivative = derivative;
            this.derivativeLoss = derivativeLoss;

            createLayer(nCount, prevoisLayer);
        }

        /**Прямой проход для нейронов слоя (вычисление)
         * @return - вектор выходных данных*/
        public double[] forward(){
            double[] output = new double[neirons.size()];

            for(int i=0; i<neirons.size(); i++) {
                Neiron neiron = neirons.get(i);
                neiron.process(activation);
                output[i] = neiron.oValue;
            }

            return output;
        }

        /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения
         * @param target - вектор целевых значений выходных нейронов
         * @param optimaizer - используемый при обучении оптимизатор весовых коэффициентов
         * @param isBatchFlg - если истина то обучение в пакетном режиме. Иначе нет
         * @param endBatchFlg - Только для пакетного режима обучения. Если истина, то данный батч закончился и необходимо скорректировать веса
         * @param batchCurrentSize - Только для пакетного режима обучения. Текущий размер пачки*/
        public void backward(double[] target, IWeightOptimaizer optimaizer, boolean isBatchFlg, boolean endBatchFlg, int batchCurrentSize){
            for(int i=0; i<target.length; i++){
                Neiron neiron = neirons.get(i);

                //вычислим дельту ошибки для нейронов выходного слоя на основе разницы между ожидаемым и полученным значениями
                neiron.calcDelta(derivative, target[i], derivativeLoss);

                //корректируем вес смещения для нейрона
                neiron.bias.process(neiron.delta, optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize); //производная по bias равна delta (так как bias влияет напрямую на iValue)

                //корректируем веса входящих связей для нейрона
                neiron.correctWeightInputLink(optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize);
            }
        }

        public static class sigmoid extends output{ public sigmoid(int nCount, Layer prevoisLayer) {super(nCount, Fun.INIT_XAVIER(prevoisLayer.neirons.size(), nCount), prevoisLayer, Fun.SIGMOID, Fun.SIGMOID_DERIVATIVE, Fun.LOSS_DERIVATIVE_MSE);}}
        public static class tanh extends output{ public tanh(int nCount, Layer prevoisLayer) {super(nCount, Fun.INIT_XAVIER(prevoisLayer.neirons.size(), nCount), prevoisLayer, Fun.TANH, Fun.TANH_DERIVATIVE, Fun.LOSS_DERIVATIVE_MSE);}}
        public static class relu extends output{ public relu(int nCount, Layer prevoisLayer) {super(nCount, Fun.INIT_HE(prevoisLayer.neirons.size()), prevoisLayer, Fun.RELU, Fun.RELU_DERIVATIVE, Fun.LOSS_DERIVATIVE_MSE);}}
        public static class softmaxAndCrossEntity extends output{
            /*Функция активации - softmax
            * функция потерь - cross entity*/
            public softmaxAndCrossEntity(int nCount, Layer prevoisLayer) {
                // Передаём фиктивные функции активации и производной,
                // они не будут использоваться, так как forward и calcDelta переопределены.
                // Инициализация весов — Xavier (подходит для softmax).
                super(nCount, Fun.INIT_XAVIER(prevoisLayer.neirons.size(), nCount), prevoisLayer, 
                        x->x,  // фиктивная активация
                        neiron->1.0,          // фиктивная производная функции активации
                        (neiron, target )-> neiron.oValue - target    // Для комбинации softmax + кросс-энтропия дельта равна (output - target) но только если целевой вектор это one-hot вектор
                );
            }

            @Override
            public double[] forward() {
                int n = neirons.size();
                double[] iValues = new double[n];

                // 1. Вычисляем взвешенные суммы (Z = W·X + b) для всех нейронов
                for (int i = 0; i < n; i++) {
                    Neiron neiron = neirons.get(i);
                    double sum = neiron.bias.item;
                    for (Link link : neiron.iLinks) {
                        sum += link.iNeiron.oValue * link.weight.item;
                    }
                    neiron.iValue = sum;  // сохраняем для возможного использования
                    iValues[i] = sum;
                }

                // 2. Применяем softmax с численной стабилизацией (вычитание максимума)
                double max = Double.NEGATIVE_INFINITY;
                for (double v : iValues) {
                    if (v > max) max = v;
                }

                double sumExp = 0.0;
                double[] expVals = new double[n];
                for (int i = 0; i < n; i++) {
                    expVals[i] = Math.exp(iValues[i] - max);
                    sumExp += expVals[i];
                }

                double[] outputs = new double[n];
                for (int i = 0; i < n; i++) {
                    double softmaxOut = expVals[i] / sumExp;
                    neirons.get(i).oValue = softmaxOut;
                    outputs[i] = softmaxOut;
                }
                return outputs;
            }
        }
    }
}
