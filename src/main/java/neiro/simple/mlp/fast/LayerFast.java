package neiro.simple.mlp.fast;

import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;
import neiro.simple.mlp.stabile.AsyncNeirons;
import neiro.simple.mlp.stabile.WeightWrapper;
import neiro.simple.mlp.stabile.model.Fun;
import neiro.simple.mlp.stabile.train.IWeightOptimaizer;

import java.io.Serializable;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**Быстрый слой*/
@FieldDefaults(level = AccessLevel.PUBLIC)
public abstract class LayerFast {
    int size; //размер текущего слоя
    LayerFast next = null; //следующий слой
    LayerFast previous = null; //предыдущий слой

    //neiron data
    WeightWrapper[] bias; //смещения

    double[] iValue;  //входные значения нейронов слоя
    double[] oValue;  //выходные значения нейронов слоя
    double[] delta;   //дельты ошибки нейронов слоя

    //связи между нейронами данного слоя и нейронами следующего слоя
    WeightWrapper[][] oLink; //[идекс-текущего-нейрона][индекс-нейрона-из-следующего-слоя]

    /**Конструктор
     * @param size - кол-во нейронов в текущем слое*/
    public LayerFast(int size) {this.size = size;}


    /**Входной слой*/
    @FieldDefaults(level = AccessLevel.PUBLIC)
    public static class input extends LayerFast{
        /**Конструктор
         * @param size - кол-во нейронов в текущем слое*/
        public input(int size){
            super(size);
            oValue= new double[size];
        }

        /**Прямой проход для нейронов слоя (вычисление)
         * @param inputValue - вектор входных данных*/
        public void forward(double[] inputValue){
            this.oValue = inputValue;
        }
    }

    /**Общий предок для промежуточного и выходного слоя*/
    public static abstract class noInput extends LayerFast{
        UnaryOperator<Double> activation; //функция активации
        BiFunction<Double, Double, Double> derivative; //производная функции активации (входное-значенией-нейрона, выходное-значение-нейрона) -> значение производной
        Supplier<Double> initFun; //функция первичной инициализации весов перед обучением

        /**Конструктор
         * @param previous - ссылка на предыдущий слой
         * @param size - кол-во нейронов в текущем слое
         * @param initFun - функция первичной инициализации весов перед обучением
         * @param activation - функция активации нейронов слоя
         * @param derivative - производная функции активации нейронов слоя*/
        public noInput(LayerFast previous, int size, Supplier<Double> initFun, UnaryOperator<Double> activation, BiFunction<Double, Double, Double> derivative) {
            super(size);
            this.initFun = initFun;
            this.activation = activation;
            this.derivative = derivative;

            this.previous = previous;

            //создадим все необходимые объекты слоя
            bias= new WeightWrapper[size];

            iValue= new double[size];
            oValue= new double[size];
            delta= new double[size];

            //инициализируем некоторые объекты предыдущего слоя которые смотрят на текущий слой
            previous.next = this;
            previous.oLink = new WeightWrapper[previous.size][size];
        }

        /**Инициализация веса
         * @param createWeightWrapperFun - функция создания WeightWrapper*/
        public void init(Function<Double, WeightWrapper> createWeightWrapperFun){
            for(int nIndex=0; nIndex<size; nIndex++) {
                bias[nIndex] = createWeightWrapperFun.apply(initFun.get());

                for (int j = 0; j < previous.size; j++)
                    previous.oLink[j][nIndex] = createWeightWrapperFun.apply(initFun.get());
            }
        }

        /**Вычисление для нейрона с индексом index*/
        protected void neironProcess(int index){
            iValue[index] = bias[index].item;
            for(int i=0; i< previous.size; i++)
                iValue[index]+= previous.oValue[i] * previous.oLink[i][index].item;

            oValue[index] = activation.apply(iValue[index]);
        }

        /**Корректируем веса входящих связей для нейрона на основе ранее вычисленной дельты нейрона
         * @param index - индекс нейрона
         * @param optimaizer - используемый при обучении оптимизатор весовых коэффициентов
         * @param isBatchFlg - если истина то обучение в пакетном режиме. Иначе нет
         * @param endBatchFlg - Только для пакетного режима обучения. Если истина, то данный батч закончился и необходимо скорректировать веса
         * @param batchCurrentSize - Только для пакетного режима обучения. Текущий размер пачки*/
        public void correctWeightInputLink(int index, IWeightOptimaizer optimaizer, boolean isBatchFlg, boolean endBatchFlg, int batchCurrentSize){
            for(int i=0; i< previous.size; i++){
                double grad = previous.oValue[i] * delta[index];
                previous.oLink[i][index].process(grad, optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize);
            }
        }

        /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения
         * @param target - вектор целевых значений выходных нейронов (для выходного слоя) или пусто для промежуточного слоя
         * @param optimaizer - используемый при обучении оптимизатор весовых коэффициентов
         * @param isBatchFlg - если истина то обучение в пакетном режиме. Иначе нет
         * @param endBatchFlg - Только для пакетного режима обучения. Если истина, то данный батч закончился и необходимо скорректировать веса
         * @param batchCurrentSize - Только для пакетного режима обучения. Текущий размер пачки*/
        public void backward(double[] target, IWeightOptimaizer optimaizer, boolean isBatchFlg, boolean endBatchFlg, int batchCurrentSize){
            AsyncNeirons.asyncIndexedFor(size, nIndex ->{ //for(int nIndex=0; nIndex<size; nIndex++){
                //Протащим дельту ошибки (вычислим для нейронов текущего слоя на основе уже известных дельт ошибок нейронов следующего слоя)
                neironCalcDelta(nIndex, target);

                //корректируем вес смещения для нейрона
                bias[nIndex].process(delta[nIndex], optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize); //производная по bias равна delta (так как bias влияет напрямую на iValue)

                //корректируем веса входящих связей для нейрона
                correctWeightInputLink(nIndex, optimaizer, isBatchFlg, endBatchFlg, batchCurrentSize);
            });
        }

        /**Вычислить дельту ошибки для нейрона
         * @param index - номер нейрона в текущем слое для которого вычисляем дельту ошибки
         * @param target - целевой вектор значений*/
        protected abstract void neironCalcDelta(int index, double[] target);
    }

    /**Промежуточный слой*/
    public static class medium extends noInput{
        /**
         * Конструктор
         *
         * @param previous   - ссылка на предыдущий слой
         * @param size       - кол-во нейронов в текущем слое
         * @param initFun - функция первичной инициализации весов
         * @param activation - функция активации нейронов слоя
         * @param derivative - производная функции активации нейронов слоя
         */
        public medium(int size, Supplier<Double> initFun, LayerFast previous, UnaryOperator<Double> activation, BiFunction<Double, Double, Double> derivative) {
            super(previous, size, initFun, activation, derivative);
        }

        /**Прямой проход по нейронам слоя (вычисление)*/
        public void forward(){
            AsyncNeirons.asyncIndexedFor(size, i-> { //for (int i=0; i<size; i++)
                neironProcess(i);
            });
        }

        /**Распространить дельту ошибки для нейрона
         * @param index - номер нейрона в текущем слое для которого вычисляем дельту ошибки на основе уже известных дельт ошибок нейронов следующего слоя с которыми связан текущий нейрон*/
        protected void neironCalcDelta(int index, double[] target){
            double sum = 0F;
            for(int i=0; i<next.size; i++)
                sum+=next.delta[i] * oLink[index][i].item;
            delta[index] = derivative.apply(iValue[index], oValue[index]) * sum;
        }

        public static class sigmoid extends LayerFast.medium { public sigmoid(int nCount, LayerFast prevoisLayer) {super(nCount, Fun.INIT_XAVIER(prevoisLayer.size, nCount), prevoisLayer, Fun.SIGMOID, Fun.SIGMOID_DERIVATIVE);}}
        public static class tanh extends LayerFast.medium { public tanh(int nCount, LayerFast prevoisLayer) {super(nCount, Fun.INIT_XAVIER(prevoisLayer.size, nCount), prevoisLayer, Fun.TANH, Fun.TANH_DERIVATIVE);}}
        public static class relu extends LayerFast.medium { public relu(int nCount, LayerFast prevoisLayer) {super(nCount, Fun.INIT_HE(prevoisLayer.size), prevoisLayer, Fun.RELU, Fun.RELU_DERIVATIVE);}}
    }

    /**Выходной слой*/
    public static class output extends noInput{
        BiFunction<Double, Double, Double> derivativeLoss; //производная функции потерь

        /**
         * Конструктор
         *
         * @param previous   - ссылка на предыдущий слой
         * @param size       - кол-во нейронов в текущем слое
         * @param initFun - функция первичной инициализации весов
         * @param activation - функция активации нейронов слоя
         * @param derivative - производная функции активации нейронов слоя
         * @param derivativeLoss - производная функции потерь
         */
        public output(int size, Supplier<Double> initFun, LayerFast previous, UnaryOperator<Double> activation, BiFunction<Double, Double, Double> derivative, BiFunction<Double, Double, Double> derivativeLoss) {
            super(previous, size, initFun, activation, derivative);
            this.derivativeLoss = derivativeLoss;
        }

        /**Прямой проход по нейронам слоя (вычисление)
         * @return - выходной вектор*/
        public double[] forward(){
            AsyncNeirons.asyncIndexedFor(size, i-> { //for (int i=0; i<size; i++)
                neironProcess(i);
            });
            return oValue;
        }

        /**Вычислить дельту ошибки для выходного нейрона
         * @param index - нейрон для которого вычисляем дельту ошибки на основе разницы между ожидаемым и полученным значениями
         * @param target - целевое(ожидаемое) значение нейрона*/
        protected void neironCalcDelta(int index, double[] target){
            //дельта ошибки на выходе нейрона = производная функции потерь * производная функции активации
            delta[index] = derivativeLoss.apply(oValue[index], target[index]) * derivative.apply(iValue[index], oValue[index]);
        }

        public static class sigmoid extends LayerFast.output { public sigmoid(int nCount, LayerFast prevoisLayer) {super(nCount, Fun.INIT_XAVIER(prevoisLayer.size, nCount), prevoisLayer, Fun.SIGMOID, Fun.SIGMOID_DERIVATIVE, Fun.LOSS_DERIVATIVE_MSE);}}
        public static class tanh extends LayerFast.output { public tanh(int nCount, LayerFast prevoisLayer) {super(nCount, Fun.INIT_XAVIER(prevoisLayer.size, nCount), prevoisLayer, Fun.TANH, Fun.TANH_DERIVATIVE, Fun.LOSS_DERIVATIVE_MSE);}}
        public static class relu extends LayerFast.output { public relu(int nCount, LayerFast prevoisLayer) {super(nCount, Fun.INIT_HE(prevoisLayer.size), prevoisLayer, Fun.RELU, Fun.RELU_DERIVATIVE, Fun.LOSS_DERIVATIVE_MSE);}}
        public static class softmaxAndCrossEntity extends LayerFast.output {
            /*Функция активации - softmax
             * функция потерь - cross entity*/
            public softmaxAndCrossEntity(int nCount, LayerFast prevoisLayer) {
                // Передаём фиктивные функции активации и производной,
                // они не будут использоваться, так как forward и calcDelta переопределены.
                // Инициализация весов — Xavier (подходит для softmax).
                super(nCount, Fun.INIT_XAVIER(prevoisLayer.size, nCount), prevoisLayer,
                        (UnaryOperator<Double> & Serializable) x->x,  // фиктивная активация
                        (BiFunction<Double, Double, Double> & Serializable) (iValue, oValue)->1.0,          // фиктивная производная функции активации
                        (BiFunction<Double, Double, Double> & Serializable) (oValue, target )-> oValue - target    // Для комбинации softmax + кросс-энтропия дельта равна (output - target) но только если целевой вектор это one-hot вектор
                );
            }

            @Override
            public double[] forward() {
                // 1. Вычисляем взвешенные суммы (Z = W·X + b) для всех нейронов
                for (int index = 0; index < size; index++) {
                    double sum = bias[index].item;
                    for(int i=0; i< previous.size; i++)
                        sum+=previous.oValue[i] * previous.oLink[i][index].item;
                    iValue[index] = sum;
                }

                // 2. Применяем softmax с численной стабилизацией (вычитание максимума)
                double max = Double.NEGATIVE_INFINITY;
                for (double v : iValue)
                    if (v > max) max = v;

                double sumExp = 0.0;
                double[] expVals = new double[size];
                for (int i = 0; i < size; i++) {
                    expVals[i] = Math.exp(iValue[i] - max);
                    sumExp += expVals[i];
                }

                for (int i = 0; i < size; i++) {
                    double softmaxOut = expVals[i] / sumExp;
                    oValue[i] = softmaxOut;
                }
                return oValue;
            }
        }

    }
}
