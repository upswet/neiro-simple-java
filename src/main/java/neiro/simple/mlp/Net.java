package neiro.simple.mlp;

import lombok.*;
import lombok.experimental.Accessors;
import lombok.experimental.FieldDefaults;
import lombok.extern.slf4j.Slf4j;

import java.io.Serializable;
import java.util.List;

/**Максимально ускоренная полносвязанная нейросеть без использования матриц, с условным разбиением на нейроны, связи и слои*/
@Slf4j
public class Net implements Serializable {

    interface FloatUnaryOp extends Serializable { float apply(float x);}
    interface FloatBinaryOp extends Serializable { float apply(float a, float b);}

    // Описание слоёв чтобы было удобно задавать нейросеть
    @AllArgsConstructor
    public static abstract class Layer {
        final Integer size;
    }
    public static class LayerInput extends Layer {
        public LayerInput(Integer size) { super(size); }
    }
    public static class LayerMedium extends Layer {
        final InitFunTypeEnum initFun;
        final ActivationTypeEnum fun;
        public LayerMedium(Integer size, InitFunTypeEnum initFun, ActivationTypeEnum fun) {
            super(size);
            this.initFun = initFun;
            this.fun = fun;
        }
    }

    // === Функции потерь (и их производные) ===
    @AllArgsConstructor
    public static enum LossTypeEnum {
        MSE((o, t) -> o - t),
        /*Mean Squared Error (MSE) — среднеквадратичная ошибка. Производная пропорциональна ошибке (чем больше ошибка, тем сильнее градиент)*/

        MAE((o, t) -> o < t ? -1f : 1f),
        /*//Mean Absolute Error (MAE) — средняя абсолютная ошибка. Производная не зависит от величины ошибки, только от знака */

        CROSS_ENTROPY((o, t) -> o - t); // Для комбинации softmax (sigmoid) + кросс-энтропия дельта равна (output - target) но только если целевой вектор это one-hot вектор)
          /*
        Категориальная кросс-энтропия (Categorical Cross-Entropy) - мера различия между двумя распределениями вероятностей. В обучении — штрафует, когда предсказанное распределение (выход softmax/сигмоиды) расходится с истинным (one‑hot или бинарной меткой). Чем меньше cross‑entropy, тем точнее предсказание

        Чаще всего используется вместе с softmax на выходном слое
        Почему именно её:
            Softmax выдаёт распределение вероятностей по классам.
            Кросс-энтропия измеряет разницу между предсказанным распределением и истинным (one-hot) распределением.
            Вместе они образуют устойчивую комбинацию, которая хорошо работает для задач многоклассовой классификации.

         Производная (для многоклассовой классификации softmax + cross‑entropy):
            По логитам (входам softmax) производная = pᵢ − yᵢ, где pᵢ — предсказанная вероятность класса i, yᵢ — истинная метка (0 или 1). Это одно из главных преимуществ комбинации — простая и устойчивая производная.
        */

        public final FloatBinaryOp derivative; //производная функции потерь: (выходное-значение-нейрона, целевое-значение-нейрона) -> потеря
    }

    // === Функции активации (примитивные) ===
    @AllArgsConstructor
    public static enum ActivationTypeEnum {
        IDENTITY(
                x -> x,
                (i, o) -> 1f
        ),
        /*без преобразования*/

        SIGMOID(
                x -> (float) (1.0 / (1.0 + Math.exp(-x))),
                (i, o) -> o * (1 - o)
                //SIGMOID.apply(iValue) * (1 - SIGMOID.apply(iValue)) - классический вариант, через входное значение нейрона
                //oValue * (1 - oValue) // ускоренный вариант - через выходное значение нейрона
        ),

        TANH(
                x -> (float) Math.tanh(x),
                (i, o) -> 1f - o * o // производная tanh = 1 - tanh²(x)
        ),

        RELU(
                x -> x > 0 ? x : 0.01f * x,
                (i, o) -> i > 0 ? 1f : 0.01f)
        ,

        SOFTMAX(
                x -> x,
                (i, o) -> 1f)
        ; // активация-заглушка, softmax применяется отдельно

        public final FloatUnaryOp activation; //функция активации: (входное-значение-нейрона) -> выходное-значение-нейрона
        public final FloatBinaryOp derivative; //производная функции активации: (входное-значенией-нейрона, выходное-значение-нейрона) -> значение производной
    }

    // === Оптимизаторы ===
    public static enum OptimizatorTypeEnum{//Перечисление поддерживаемых оптимизаторов весовых коэффициентов
        SSG,
        /*Стандарный градиентный спуск*/

        MOMENTUM,
        /*Метод момента*/

        NAG,
        /*Модифицированный метод момента*/

        ADAM;
        /*Адам - оптимизатор*/
    }

    /**Гиперпараметры для оптимизатора весовых коэффициентов*/
    @Accessors(chain = true)
    @NoArgsConstructor
    @Setter
    public static class Optimizator {
        OptimizatorTypeEnum type;

        /**Гиперпараметры оптимизатора SSG*/
        float learningRate; //коэффициент скорости обучения
        /**Гиперпараметры оптимизатора MOMENTUM, NAG*/
        float momentumCoef; //коэффициент момента (коэф инерции)
        /**Гиперпараметры оптимизатора ADAM*/
        float beta1 = 0.9F;
        float beta2 = 0.999F;
        float epsilon = 1e-8F;
        /**Параметры для ускорения расчёта*/
        float beta1Pow;
        float beta2Pow;
        float invCorrection1;
        float invCorrection2;

        public Optimizator init() {
            if (type == null) throw new RuntimeException("Not specified type!");
            if (learningRate == 0) throw new RuntimeException("Not specified learningRate!");

            switch (type){
                case MOMENTUM, NAG -> {if (momentumCoef == 0) throw new RuntimeException("Not specified momentumCoef!");}
                case ADAM -> {
                    //задаём значения для первого шага
                    adamInit();
                }
            }

            return this;
        }

        private void adamInit() {
            beta1Pow = 1.0F;
            beta2Pow = 1.0F;
            invCorrection1 = 1.0F / (1.0F - beta1); // для первого шага
            invCorrection2 = 1.0F / (1.0F - beta2);
        }

        public void adamNextStep(){
            beta1Pow *= beta1;
            beta2Pow *= beta2;
            invCorrection1 = 1.0F / (1.0F - beta1Pow);
            invCorrection2 = 1.0F / (1.0F - beta2Pow);
        }
        public void adamReset(){
            beta1Pow = beta1;
            beta2Pow = beta2;
            invCorrection1 = 1.0F / (1.0F - beta1Pow);
            invCorrection2 = 1.0F / (1.0F - beta2Pow);
        };
    }

    @AllArgsConstructor
    public static enum InitFunTypeEnum {//перечисление функций инициализации весовых коэффициентов
        NORMAL(
                (fanIn, fanOut) -> NetUtils.initNormal(0.01f)
        ),
        /*Нормальное распределение (Гаусса)*/

        XAVIER((fanIn, fanOut) -> {
            double std = Math.sqrt(2.0 / (fanIn + fanOut));
            return NetUtils.initNormal(std);
        }),
        /*Распредление хавьера (Для кого: tanh, sigmoid, softmax (симметричные, с насыщением))*/

        HE((fanIn, fanOut) -> {
            double std = Math.sqrt(2.0 / fanIn);
            return NetUtils.initNormal(std);
        });
        /*Распредлеение Хе (Для кого: ReLU, Leaky ReLU, PReLU (несимметричные, с отсечением отрицательных значений).
         * Учитывает, что ReLU зануляет ~50% нейронов, поэтому дисперсию нужно увеличить вдвое по сравнению с Xavier*/

        public final FloatBinaryOp init; // (кол-нейронов-в-пред-слое, кол-нейронов-в-этом-слое) -> вес
    }

    //Входные данные для обучения
    public record Example(float[] input, float[] target) {}
    public record Examples(Example[] trainData, Example[] testData) {}

    //Результаты обучения
    @FieldDefaults(level = AccessLevel.PUBLIC)
    public static class TrainResult {
        public Long duration = 0L;
        public int epoch = 0;
        public float quality = 0f;
        public void set(Long duration, int epoch, float quality) {
            this.duration = duration;
            this.epoch = epoch;
            this.quality = quality;
        }
        @Override
        public String toString() {
            return "TrainResult{duration=" + duration + ", epoch=" + epoch + ", quality=" + quality + '}';
        }
    }

    /**Интерфейс для тестирования нейросети (функция оценки)*/
    public interface Estimation {
        /**
         * Проверка результата вычислений во время тестирования
         *
         * @param expectedVec     - вектор ожидаемых значений
         * @param outputVec       - вектор выходных значений
         * @param acceptableError - допустимая ошибка при которой ответ считается верным
         * @return - правильный ответ или нет
         */
        boolean process(float[] outputVec, float[] expectedVec, Float acceptableError);
    }
    /**Функция эстимации. По максимальному значению*/
    public static Estimation estimationMax = (outputVec, expectedVec, acceptableError) ->
            NetUtils.findMax(outputVec).equals(NetUtils.findMax(expectedVec));
    /**Функция эстимации. По разнице между векторами*/
    public static Estimation estimationLoss = (outputVec, expectedVec, acceptableError) -> {
        double loss = 0;
        for (int i = 0; i < outputVec.length; i++)
            loss += Math.abs(expectedVec[i] - outputVec[i]);
        return loss < acceptableError;
    };

    // === Поля сети ===
    private final int[] sizes;                 // размеры слоёв (кол-во нейронов)
    private final ActivationTypeEnum[] funs;   // тип активации каждого слоя
    private final LossTypeEnum lossType; //тип функции потерь

    // нейроны
    private transient float[][] iValue;   // входы нейронов
    private transient float[][] oValue;   // выходы нейронов
    private transient float[][] delta;    // дельты ошибок

    // смещения (bias)
    private final float[][] biasWeight;
    private transient float[][] biasAcc;   // для batch
    private transient float[][] biasM1;    // для моментов
    private transient float[][] biasM2;

    // веса связей: [layer][currNeuron][prevNeuron]
    private final float[][][] linkWeight;
    private transient float[][][] linkAcc;
    private transient float[][][] linkM1;
    private transient float[][][] linkM2;

    // === Конструктор ===
    public Net(List<Layer> layers, @NonNull LossTypeEnum lossType) {
        //Проверки
        if (layers.size() < 2) throw new RuntimeException("layer.size < 2 !");
        if (!(layers.getFirst() instanceof LayerInput))
            throw new RuntimeException("First layer not input!");
        if (lossType == LossTypeEnum.CROSS_ENTROPY) {
            LayerMedium outputLayer = (LayerMedium) layers.getLast();
            if (outputLayer.fun != ActivationTypeEnum.SIGMOID && outputLayer.fun != ActivationTypeEnum.SOFTMAX)
                throw new RuntimeException("For CROSS_ENTROPY only SOFTMAX or SIGMOID allowed");
        }
        this.lossType = lossType;

        int layerCount = layers.size();
        sizes = new int[layerCount];
        iValue = new float[layerCount][];
        oValue = new float[layerCount][];
        delta = new float[layerCount][];
        biasWeight = new float[layerCount][];
        biasAcc = new float[layerCount][];
        biasM1 = new float[layerCount][];
        biasM2 = new float[layerCount][];
        linkWeight = new float[layerCount][][];
        linkAcc = new float[layerCount][][];
        linkM1 = new float[layerCount][][];
        linkM2 = new float[layerCount][][];
        funs = new ActivationTypeEnum[layerCount];

        for (int i = 0; i < layerCount; i++) {
            sizes[i] = layers.get(i).size;
            oValue[i] = new float[sizes[i]];
            if (i > 0) {
                LayerMedium layer = (LayerMedium) layers.get(i);
                funs[i] = layer.fun;
                iValue[i] = new float[sizes[i]];
                delta[i] = new float[sizes[i]];
                biasWeight[i] = new float[sizes[i]];
                biasAcc[i] = new float[sizes[i]];
                biasM1[i] = new float[sizes[i]];
                biasM2[i] = new float[sizes[i]];

                // инициализация смещений
                for (int j = 0; j < sizes[i]; j++) {
                    biasWeight[i][j] = layer.initFun.init.apply(sizes[i-1], sizes[i]);
                }

                // инициализация весов связей: [curr][prev]
                linkWeight[i] = new float[sizes[i]][sizes[i-1]];
                linkAcc[i] = new float[sizes[i]][sizes[i-1]];
                linkM1[i] = new float[sizes[i]][sizes[i-1]];
                linkM2[i] = new float[sizes[i]][sizes[i-1]];
                for (int curr = 0; curr < sizes[i]; curr++) {
                    for (int prev = 0; prev < sizes[i-1]; prev++) {
                        linkWeight[i][curr][prev] = layer.initFun.init.apply(sizes[i-1], sizes[i]);
                    }
                }
            }
        }
    }

    /**Прямой проход. Вычисляет вектор ответа на основе входного вектора
     * @param input - входной вектор
     * @return - выходной вектор*/
    public float[] forward(float[] input) {
        // Задать входной вектор через нейроны входного слоя
        oValue[0] = input;

        //для каждого слоя после первого начнём процесс прямого распространения сигнала
        for (int layer = 1; layer < sizes.length; layer++) {
            float[] iVal = iValue[layer];
            float[] bias = biasWeight[layer];
            float[][] link = linkWeight[layer];       // [curr][prev]
            float[] oPrev = oValue[layer-1];
            int prevSize = sizes[layer-1];
            int currSize = sizes[layer];
            ActivationTypeEnum act = funs[layer];

            //для каждого нейрона вычислим входной сигнал как сумму выходных сигналов нейронов предыдущего слоя умножимых на веса связей от них к текущему нейрону с учётом его смещения
            for (int curr = 0; curr < currSize; curr++) {
                float sum = bias[curr];
                float[] linkCurr = link[curr];
                for (int prev = 0; prev < prevSize; prev++) {
                    sum += oPrev[prev] * linkCurr[prev];
                }
                iVal[curr] = sum;

                //вычислим выходное значение нейрона через функцию его активации от его входного значения
                oValue[layer][curr] = act.activation.apply(sum);
            }
        }

        // softmax для выходного слоя, если нужно
        if (funs[sizes.length-1] == ActivationTypeEnum.SOFTMAX) {
            oValue[sizes.length-1] = NetUtils.toSoftmax(oValue[sizes.length-1]);
        }

        return oValue[sizes.length-1];
    }

    /**Обратное распространение ошибки. Обновляет весовые коэффициенты (обучает сеть) после прямого прохода
     * @param target - целевой вектор ответа
     * @param optimizator - параметры оптимизатора весовых коэффициентов
     * @param isBatch - если истина, то пакетный режим, иначе нет*/
    public void backward(float[] target, Optimizator optimizator, boolean isBatch) {
            /*
        Ошибка вычисляется на выходном слое.
        Затем она распространяется назад: от выходного слоя к входному.
        Каждый нейрон (кроме входных) имеет дельту (δ), которая показывает, насколько он виноват в ошибке.

        Layer.output.backward:
            - вычисляет дельту для каждого нейрона выходного слоя на основе разницы между целевыми и полученными данными
            - обновляет веса входящих связей (iLinks) этих нейронов.
        Layer.medium.backward:
            - вычисляет дельту для каждого нейрона скрытого слоя (используя дельты следующего слоя и веса исходящих связей).
            - обновляет веса входящих связей (iLinks) этих нейронов.
        Layer.input.backward:
            - Входной слой не имеет входящих связей, а его исходящие связи (oLinks) уже обновлены на предыдущем шаге. Ничего не делаем
        */

        //для всех слоёв кроме первого, двигаясь задом наперёд, вычислим дельту ошибки и скорректируем веса смещений и веса связей
        for (int layer = sizes.length - 1; layer >= 1; layer--) {
            float[] iVal = iValue[layer];
            float[] oVal = oValue[layer];
            float[] deltaLayer = delta[layer];
            ActivationTypeEnum act = funs[layer];
            int currSize = sizes[layer];

            //для всех нейронов данного слоя
            for (int curr = 0; curr < currSize; curr++) {
                //вычислим дельту ошибки
                if (layer == sizes.length - 1) {  // это выходной слой
                    //Если это выходной слой, то вычислим дельту ошибки как разницу между целевым (ожидаемым) вектором ответа и тем вектором который выдала нейросеть как (в общем виде)
                    //дельта ошибки на нейроне выходного слоя = производная функции потерь * производная функции активации
                    deltaLayer[curr] = lossType.derivative.apply(oVal[curr], target[curr]) * act.derivative.apply(iVal[curr], oVal[curr]);
                } else {                          // это скрытый слой
                    //Если это промежуточный слой, то вычислим дельту ошибки на основе уже известных дельт ошибок нейронов следующего слоя с которыми связан текущий нейрон
                    //дельта ошибки на нейроне промежуточного слоя = сумма взвешенных дельт ошибок нейронов следующего слоя связанных с данным * производная функции активации
                    float sum = 0f;
                    float[] deltaNext = delta[layer+1];
                    float[][] linkNext = linkWeight[layer+1]; // [nextCurr][curr]
                    int nextSize = sizes[layer+1];
                    for (int next = 0; next < nextSize; next++) {
                        sum += deltaNext[next] * linkNext[next][curr];
                    }
                    deltaLayer[curr] = sum * act.derivative.apply(iVal[curr], oVal[curr]);
                }

                //зная дельту ошибки нейрона
                if (!isBatch){  //это не пакетный режим
                    //сразу скорректируем вес смещения для нейрона и входящих в него связей
                    updateWeightNoBatch(optimizator, layer, curr, delta[layer][curr]);
                }else {         //это пакетный режим
                    // накопим градиенты смещения и входящих в нейрон связей чтобы обновить их в конце батча в методе trainEpoch

                    // Накопление градиента для bias
                    biasAcc[layer][curr] += deltaLayer[curr];//производная по bias равна delta (так как bias влияет напрямую на iValue)

                    // Накопление градиентов для связей
                    float[] oPrev = oValue[layer - 1];
                    int prevSize = sizes[layer - 1];
                    for (int prev = 0; prev < prevSize; prev++) {
                        linkAcc[layer][curr][prev] += oPrev[prev] * deltaLayer[curr];
                    }
                }
            }
        }
    }

    /**Обновить веса смещения и входящих связей для конкретного нейрона через его дельту ошибки без пакетного режима
     * @param optimizator - параметры оптимизатора весового коэффициента
     * @param layer - номер слоя
     * @param curr - номер нейрона в слое*/
    private void updateWeightNoBatch(Optimizator optimizator, int layer, int curr, float deltaCurr) {
        switch (optimizator.type) {
            case SSG -> {
                //обновим bias (смещения)
                biasWeight[layer][curr] -= optimizator.learningRate * deltaCurr; //производная по bias равна delta (так как bias влияет напрямую на iValue) grad =
                //обновим веса входящих связей
                float[] oPrev = oValue[layer - 1];
                float[] linkCurr = linkWeight[layer][curr];
                int prevSize = sizes[layer - 1];
                for (int prev = 0; prev < prevSize; prev++) {
                    linkCurr[prev] -= optimizator.learningRate * oPrev[prev] * deltaCurr;
                }
            }
            case MOMENTUM -> {
                //обновим bias (смещения)
                biasM1[layer][curr] = biasM1[layer][curr] * optimizator.momentumCoef + optimizator.learningRate * deltaCurr; //производная по bias равна delta (так как bias влияет напрямую на iValue)
                biasWeight[layer][curr] -= biasM1[layer][curr];
                //обновим веса входящих связей
                float[] oPrev = oValue[layer - 1];
                float[] linkCurr = linkWeight[layer][curr];
                float[] linkCurrM1 = linkM1[layer][curr];
                int prevSize = sizes[layer - 1];
                for (int prev = 0; prev < prevSize; prev++) {
                    linkCurrM1[prev] = linkCurrM1[prev] * optimizator.momentumCoef + optimizator.learningRate * oPrev[prev] * deltaCurr;
                    linkCurr[prev] -= linkCurrM1[prev];
                }
            }
            case NAG -> {
                //обновим bias (смещения)
                float prevM = biasM1[layer][curr];// Сохраняем предыдущее значение момента (до обновления)
                biasM1[layer][curr] = biasM1[layer][curr] * optimizator.momentumCoef + optimizator.learningRate * deltaCurr; //производная по bias равна delta (так как bias влияет напрямую на iValue)
                biasWeight[layer][curr] -= ((1 + optimizator.momentumCoef) * biasM1[layer][curr] - optimizator.momentumCoef * prevM); // Обновляем вес по формуле NAG: w = w - ((1 + μ) * m_new - μ * m_old)
                //обновим веса входящих связей
                float[] oPrev = oValue[layer - 1];
                float[] linkCurr = linkWeight[layer][curr];
                float[] linkCurrM1 = linkM1[layer][curr];
                int prevSize = sizes[layer - 1];
                for (int prev = 0; prev < prevSize; prev++) {
                    prevM = linkCurrM1[prev]; // Сохраняем предыдущее значение момента (до обновления)
                    linkCurrM1[prev] = linkCurrM1[prev] * optimizator.momentumCoef + optimizator.learningRate * oPrev[prev] * deltaCurr;
                    linkCurr[prev] -= ((1 + optimizator.momentumCoef) * linkCurrM1[prev] - optimizator.momentumCoef * prevM); // Обновляем вес по формуле NAG: w = w - ((1 + μ) * m_new - μ * m_old)
                }
            }
            case ADAM -> {
                //обновим bias (смещения)
                // Обновление моментов Adam
                biasM1[layer][curr] = optimizator.beta1 * biasM1[layer][curr] + (1 - optimizator.beta1) * deltaCurr; //производная по bias равна delta (так как bias влияет напрямую на iValue)
                biasM2[layer][curr] = optimizator.beta2 * biasM2[layer][curr] + (1 - optimizator.beta2) * deltaCurr * deltaCurr;
                // Смещение моментов
                double mHat = biasM1[layer][curr] * optimizator.invCorrection1;
                double vHat = biasM2[layer][curr] * optimizator.invCorrection2;
                //  Обновление веса
                biasWeight[layer][curr] = (float) (biasWeight[layer][curr] - optimizator.learningRate * mHat / (Math.sqrt(vHat) + optimizator.epsilon));

                //обновим веса входящих связей
                float[] oPrev = oValue[layer - 1];
                float[] linkCurr = linkWeight[layer][curr];
                float[] linkCurrM1 = linkM1[layer][curr];
                float[] linkCurrM2 = linkM2[layer][curr];
                int prevSize = sizes[layer - 1];
                for (int prev = 0; prev < prevSize; prev++) {
                    float grad = oPrev[prev] * deltaCurr;
                    // Обновление моментов Adam
                    linkCurrM1[prev] = optimizator.beta1 * linkCurrM1[prev] + (1 - optimizator.beta1) * grad; //производная по bias равна delta (так как bias влияет напрямую на iValue)
                    linkCurrM2[prev] = optimizator.beta2 *  linkCurrM2[prev] + (1 - optimizator.beta2) * grad * grad;
                    // Смещение моментов
                    mHat = linkCurrM1[prev] * optimizator.invCorrection1;
                    vHat = linkCurrM2[prev] * optimizator.invCorrection2;
                    //  Обновление веса
                    linkCurr[prev] = (float) (linkCurr[prev] - optimizator.learningRate * mHat / (Math.sqrt(vHat) + optimizator.epsilon));
                }
            }
        }
    }

    /**Обновить веса смещения и входящих связей на основе накопленных в пакетном режиме градиентов
     * @param opt - параметры оптимизатора весового коэффициента
     * @param batchCurrentSize -текущий размер пачки */
    private void applyGradients(Optimizator opt, int batchCurrentSize) {
        float invBatchCurrentSize = 1f / batchCurrentSize;
        for (int layer = 1; layer < sizes.length; layer++) {
            int prevSize = sizes[layer-1];
            int currSize = sizes[layer];
            for (int curr = 0; curr < currSize; curr++) {
                float avgDeltaBias = biasAcc[layer][curr] * invBatchCurrentSize;
                // Обновление bias в зависимости от типа оптимизатора
                switch (opt.type) {
                    case SSG:
                        biasWeight[layer][curr] -= opt.learningRate * avgDeltaBias;
                        break;
                    case MOMENTUM:
                        biasM1[layer][curr] = biasM1[layer][curr] * opt.momentumCoef + opt.learningRate * avgDeltaBias;
                        biasWeight[layer][curr] -= biasM1[layer][curr];
                        break;
                    case NAG:
                        float prevM = biasM1[layer][curr];
                        biasM1[layer][curr] = biasM1[layer][curr] * opt.momentumCoef + opt.learningRate * avgDeltaBias;
                        biasWeight[layer][curr] -= ((1 + opt.momentumCoef) * biasM1[layer][curr] - opt.momentumCoef * prevM);
                        break;
                    case ADAM:
                        biasM1[layer][curr] = opt.beta1 * biasM1[layer][curr] + (1 - opt.beta1) * avgDeltaBias;
                        biasM2[layer][curr] = opt.beta2 * biasM2[layer][curr] + (1 - opt.beta2) * avgDeltaBias * avgDeltaBias;
                        double mHat = biasM1[layer][curr] * opt.invCorrection1;
                        double vHat = biasM2[layer][curr] * opt.invCorrection2;
                        biasWeight[layer][curr] -= (float) (opt.learningRate * mHat / (Math.sqrt(vHat) + opt.epsilon));
                        break;
                }
                // Обновление весов связей
                for (int prev = 0; prev < prevSize; prev++) {
                    float avgGradLink = linkAcc[layer][curr][prev] * invBatchCurrentSize;
                    switch (opt.type) {
                        case SSG:
                            linkWeight[layer][curr][prev] -= opt.learningRate * avgGradLink;
                            break;
                        case MOMENTUM:
                            linkM1[layer][curr][prev] = linkM1[layer][curr][prev] * opt.momentumCoef + opt.learningRate * avgGradLink;
                            linkWeight[layer][curr][prev] -= linkM1[layer][curr][prev];
                            break;
                        case NAG:
                            float prevMLink = linkM1[layer][curr][prev];
                            linkM1[layer][curr][prev] = linkM1[layer][curr][prev] * opt.momentumCoef + opt.learningRate * avgGradLink;
                            linkWeight[layer][curr][prev] -= ((1 + opt.momentumCoef) * linkM1[layer][curr][prev] - opt.momentumCoef * prevMLink);
                            break;
                        case ADAM:
                            linkM1[layer][curr][prev] = opt.beta1 * linkM1[layer][curr][prev] + (1 - opt.beta1) * avgGradLink;
                            linkM2[layer][curr][prev] = opt.beta2 * linkM2[layer][curr][prev] + (1 - opt.beta2) * avgGradLink * avgGradLink;
                            double mHatLink = linkM1[layer][curr][prev] * opt.invCorrection1;
                            double vHatLink = linkM2[layer][curr][prev] * opt.invCorrection2;
                            linkWeight[layer][curr][prev] -= (float) (opt.learningRate * mHatLink / (Math.sqrt(vHatLink) + opt.epsilon));
                            break;
                    }
                    // Обнуляем аккумулятор связи
                    linkAcc[layer][curr][prev] = 0f;
                }
                // Обнуляем аккумулятор bias
                biasAcc[layer][curr] = 0f;
            }
        }
    }

    /**Обучить нейросетью на одном примере один раз.
     * @param data - пример для обучения
     * @param optimizator - параметры оптимизатора весовых коэффициентов
     * @param isBatch - если используем пакетный режим*/
    private void trainExample(Example data, Optimizator optimizator, boolean isBatch){
        float[] output = this.forward(data.input());
        if (!isBatch && optimizator.type.equals(OptimizatorTypeEnum.ADAM)) optimizator.adamNextStep();//если это адам оптимизатор то мы должны вызывать перед каждым обновлением весов
        backward(data.target, optimizator, isBatch);
    }

    /**Тренировка в пределах одной эпохи
     * @param data - данные для обучения
     * @param optimizator - параметры оптимизатора весовых коэффициентов
     * @param batchSize - размер пачки. Минус один если не используем пакетное обучение
     * @param periodStepPrint - периодичность (в шагах) когда выводить промежуточные результаты
     * @return - длитлеьность обучения в пределах одной эпохи*/
    public long trainEpoch(Example[] data, Optimizator optimizator, int batchSize, int periodStepPrint){
        int printCount = periodStepPrint; //сколько шагов осталось до печати промежуточных значений
        int batchCurrentSize = 0; //для пакетного режима обработки. Номер текущего примера в пакете

        long start = System.nanoTime();
        long lastStep = start;
        boolean isBatch = batchSize > 0;
        for(int i=0; i<data.length; i++, printCount--){
            if (printCount==0){
                log.info("step {} / {}. step duration {}. all duration {} ", i, data.length, (System.nanoTime()-lastStep)/ 1_000_000L, (System.nanoTime()-start) / 1_000_000L);
                printCount = periodStepPrint;
                lastStep=System.nanoTime();
            }

            trainExample(data[i], optimizator, isBatch);

            if (isBatch) { //это пакетный режим
                batchCurrentSize++;

                // если батч завершён – применяем градиенты
                if (batchCurrentSize == batchSize || i == data.length - 1) {
                    if (optimizator.type.equals(OptimizatorTypeEnum.ADAM))
                        optimizator.adamNextStep();//если это адам оптимизатор то мы должны вызывать перед каждым обновлением весов

                    applyGradients(optimizator, batchCurrentSize);
                    batchCurrentSize = 0;
                }
            }
        }
        return (System.nanoTime()-start) / 1_000_000L;
    }

    /**Протестировать эффективность нейросети на примере тестовых данных
     * @param data - данные для тестирования
     * @param estimation - функция оценки качества ответа нейросети
     * @param acceptableError - допустимая ошибка при которой ответ нейросети всё равно считается правильным
     * @return - доля правильных ответов*/
    public float test(Example[] data, Estimation estimation, Float acceptableError) {
        int success = 0;
        for (Example ex : data) {
            float[] output = forward(ex.input());
            if (estimation.process(output, ex.target(), acceptableError))
                success++;
        }
        return (float) success / data.length;
    }

    /**Обучить нейросеть
     * @param epoch - количество эпох обучения
     * @param data - данные для обучения и тестирования
     * @param estimation - функция оценки качества ответа нейросети
     * @param acceptableError - допустимая ошибка при которой ответ нейросети всё равно считается правильным
     * @param optimizator - параметры оптимизатора весовых коэффициентов
     * @param batchSize - размер пачки. Минус один если не используем пакетное обучение
     * @param periodStepPrint - периодичность (в шагах) когда выводить промежуточные результаты
     * @return - доля результаты обучения и тестирования*/
    public TrainResult train(int epoch, Examples data, Estimation estimation, Float acceptableError, Optimizator optimizator, int batchSize, int periodStepPrint) {
        TrainResult result = new TrainResult();
        for (int e = 1; e <= epoch; e++) {
            long dur = trainEpoch(data.trainData(), optimizator, batchSize, periodStepPrint);
            float qual = test(data.testData(), estimation, acceptableError);
            result.set(dur, e, qual);
            log.info(result.toString());
            if (qual >= 1 - acceptableError) {
                log.info("Required accuracy reached: {}", qual);
                break;
            }
        }
        return result;
    }
}