package neiro.simple.mlp.stabile.model;

import lombok.AccessLevel;
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
public abstract class Layer implements Serializable {
    final public List<Neiron> neirons = new ArrayList<>();

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

    /**Входной слой*/
    @FieldDefaults(level = AccessLevel.PUBLIC)
    public static class input extends Layer{
        /**Конструктор входного слоя
         * @param nCount - количество нейронов в слое*/
        public input(int nCount) {
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

    /**Общий предок для промежуточного и выходного слоёв*/
    public static abstract class noInput extends Layer{
        final Supplier<Double> initFun;
        UnaryOperator<Double> activation; //функция активации
        BiFunction<Double, Double, Double> derivative; //производная функции активации

        /**Конструктор не входного слоя
         * @param nCount - количество нейронов в слое
         * @param initFun - функция инициализации весов
         * @param prevoisLayer - предыдущий слой
         * @param activation - функция активации
         * @param derivative - производная функции активации
         * */
        public noInput(int nCount, Supplier<Double> initFun, Layer prevoisLayer, UnaryOperator<Double> activation, BiFunction<Double, Double, Double> derivative) {
            this.activation = activation;
            this.derivative = derivative;
            this.initFun = initFun;

            createLayer(nCount, prevoisLayer);
        }

        /**Инициализация веса
         * @param createWeightWrapperFun - функция создания WeightWrapper*/
        public void init(Function<Double, WeightWrapper> createWeightWrapperFun){
            for(Neiron neiron : neirons)
                neiron.init(createWeightWrapperFun, initFun);
        }

        /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения
         * @param target - вектор целевых значений выходных нейронов (для выходного слоя) или пусто для промежуточного слоя
         * @param optimaizer - используемый при обучении оптимизатор весовых коэффициентов
         * @param isBatchFlg - если истина то обучение в пакетном режиме. Иначе нет
         * @param endBatchFlg - Только для пакетного режима обучения. Если истина, то данный батч закончился и необходимо скорректировать веса
         * @param batchCurrentSize - Только для пакетного режима обучения. Текущий размер пачки*/
        public void backward(double[] target, IWeightOptimaizer optimaizer, boolean isBatchFlg, boolean endBatchFlg, int batchCurrentSize){
            for(int i=0; i<neirons.size(); i++) {
                 Neiron neiron = neirons.get(i);

                 //Если это выходной слой то вычислим дельту ошибки для выходного нейрона на основе разницы между ожидаемым и полученным значениям
                //Если это промежуточный слой то протащим дельту ошибки (вычислим для нейронов текущего слоя на основе уже известных дельт ошибок нейронов следующего слоя)
                calcDelta(neiron, target == null ? -1 : target[i]);

                //корректируем вес смещения для нейрона
                neiron.bias.process(neiron.delta, optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize); //производная по bias равна delta (так как bias влияет напрямую на iValue)

                //корректируем веса входящих связей для нейрона
                neiron.correctWeightInputLink(optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize);
            }
        }

        /**Вычислить дельту ошибки для нейрона
         * @param neiron - нейрон для которого вычисляем дельту ошибки
         * @param target - целевое(ожидаемое) значение нейрона*/
        protected abstract void  calcDelta(Neiron neiron, double target);
    }

    /**Промежуточный слой*/
    @FieldDefaults(level = AccessLevel.PUBLIC)
    public static class medium extends noInput {
        /**Конструктор промежуточного слоя
         * @param nCount - количество нейронов в слое
         * @param initFun - функция инициализации весов
         * @param prevoisLayer - предыдущий слой
         * @param activation - функция активации
         * @param derivative - производная функции активации
         * */
        public medium(int nCount, Supplier<Double> initFun, Layer prevoisLayer, UnaryOperator<Double> activation, BiFunction<Double, Double, Double> derivative) {
            super(nCount, initFun, prevoisLayer, activation, derivative);
        }

        /**Прямой проход для нейронов слоя (вычисление)*/
        public void forward(){
           for(Neiron neiron : neirons)
                neiron.process(activation);
        }

        @Override
        /**Распространить дельту ошибки (Вычислить на основе уже известных дельт ошибок нейронов следующего слоя с которыми он связан)
         * @param neiron - нейрон для которого вычисляем дельту ошибки
         * @param target - не используется. Оставлен для совместимости*/
        protected void calcDelta(Neiron neiron, double target) {
            double sum = 0F;
            for (Link outLink : neiron.oLinks)
                sum += outLink.oNeiron.delta * outLink.weight.item;
            neiron.delta = derivative.apply(neiron.iValue, neiron.oValue) * sum;
        }

        public static class sigmoid extends medium{ public sigmoid(int nCount, Layer prevoisLayer) {super(nCount, Fun.INIT_XAVIER(prevoisLayer.neirons.size(), nCount), prevoisLayer, Fun.SIGMOID, Fun.SIGMOID_DERIVATIVE);}}
        public static class tanh extends medium{ public tanh(int nCount, Layer prevoisLayer) {super(nCount, Fun.INIT_XAVIER(prevoisLayer.neirons.size(), nCount), prevoisLayer, Fun.TANH, Fun.TANH_DERIVATIVE);}}
        public static class relu extends medium{ public relu(int nCount, Layer prevoisLayer) {super(nCount, Fun.INIT_HE(prevoisLayer.neirons.size()), prevoisLayer, Fun.RELU, Fun.RELU_DERIVATIVE);}}
    }

    /**Выходной слой*/
    @FieldDefaults(level = AccessLevel.PUBLIC)
    public static class output extends noInput {
        BiFunction<Double, Double, Double> derivativeLoss; //производная функции потерь

        /**
         * Конструктор выходного слоя
         * @param nCount                 - количество нейронов в слое
         * @param initFun - функция инициализации весов
         * @param prevoisLayer           - предыдущий слой
         * @param activation             - функция активации
         * @param derivative             - производная функции активации
         * @param derivativeLoss - производная функции потерь
         */
        public output(int nCount, Supplier<Double> initFun, Layer prevoisLayer, UnaryOperator<Double> activation, BiFunction<Double, Double, Double> derivative, BiFunction<Double, Double, Double> derivativeLoss) {
            super(nCount, initFun, prevoisLayer, activation, derivative);
            this.derivativeLoss = derivativeLoss;
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

        @Override
        /**Вычислить дельту ошибки для выходного нейрона на основе разницы между ожидаемым и полученным значениями
         * @param neiron - нейрон для которого вычисляем дельту ошибки
         * @param target - целевое(ожидаемое) значение нейрона*/
        protected void calcDelta(Neiron neiron, double target) {
            //дельта ошибки на выходе нейрона = производная функции потерь * производная функции активации
            neiron.delta=derivativeLoss.apply(neiron.oValue, target) * derivative.apply(neiron.iValue, neiron.oValue);
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
                        (UnaryOperator<Double> & Serializable)x->x,  // фиктивная активация
                        (BiFunction<Double, Double, Double> & Serializable) (iValue, oValue)->1.0,          // фиктивная производная функции активации
                        (BiFunction<Double, Double, Double> & Serializable) (oValue, target )-> oValue - target    // Для комбинации softmax + кросс-энтропия дельта равна (output - target) но только если целевой вектор это one-hot вектор
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
