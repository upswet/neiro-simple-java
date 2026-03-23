package neiro.simple.mlp.old;

import lombok.*;
import lombok.experimental.Accessors;
import lombok.experimental.FieldDefaults;

import java.io.*;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**Реализация нейросети класса MLP (Полносвязнная нейроная сеть"). Каждый нейрон реализован как объект. Без матричных вычислений
 *
 * Работа сети. Прямой проход (forward propagation)
 * 1. Для нейронов входного слоя входным и выходным значением устанавливается вектор признаков Х подаваемый на вход нейросети.
 * 2. Для каждого следующего слоя вычисляем
 *      вектор входного значения j-го слоя Zj = W*X+b  где Х - вектор выходных значений из предыдущего слоя, W — матрица весов связей между нейронами предыдущего слоя и текущего, b — вектор смещений
 *      вектор выходного значения j-го слоя Y = f(Z) где f - нелинейная функция активации
 * 3. Процесс повторяется для всех слоёв, включая выходной. Выходные значения нейронов выходного слоя и есть вектор ответа Y
 *
 * Обучение MLP: обратное распространение (backpropagation). Метод обратного распространения ошибки с градиентным спуском
 * 0. Осуществляем прямой проход получая из вектора входных данных Х вектор выходных данных Y а также для каждого слоя j запоминая вектора Zj - вектор входа конкретно для j-го слоя (понятно что Z0 == X)
 * Термины:
 *    Функция потерь (ошибки) - скалярная метрика, показывающая, насколько сильно предсказание сети Y отличается от истинного значения T. Обучение это минимизация функции потерь.
 *    Дельта ошибки - частная производная функции потерь по взвешенному входу нейрона. Показывает насколько сильно изменение входа в нейрон повлияет на его ошибку.
 *
 * 1. Имея вектор правильного ответа T для нейронов выходного слоя
 * 1.1 Вычисляем дельту ошибки как частную производную функции потерь по взвешенному входу этого нейрона (dL/dz) = dL/dy * f~(z) то есть частная производная фун ошибки по выходу нейрона * производную функции его активации от взвешенного входа нейрона
 * Примеры функции потерь:
 *
 * MSE (Mean Squared Error) для задач регрессии = 1/2 * (target - output)^2 тогда её производная (output - target)
 *
 * Cross-Entropy для классификации = - сумма (t * log(y)) где суммируем по всем классифицируемым классам, t - истинное значение нейрона, а y - рассчитанное значение нейрона или softmax(z) то есть предсказанная вероятность для класса i.
 * Для случая, когда на выходном слое используется softmax в качестве функции активации, а в качестве функции потерь — кросс-энтропия (Cross-Entropy), вычисление дельты ошибки и градиентов имеет важное упрощение
 * дельта ошибки тогда будет y - t. Это справедливо при условии, что целевой вектор T нормирован (сумма его компонент равна 1, что выполняется для one-hot кодирования или сглаженных меток). Никаких дополнительных сомножителей, связанных с производной активации, здесь не требуется, так как они уже "сократились" благодаря сочетанию softmax и кросс-энтропии.
 * градиент вычисляется стандартно
 * Представленный алгоритм для выходного слоя (пункт 1.1) специфичен именно для пары softmax + кросс-энтропия. Если бы на выходном слое использовалась другая активация (например, сигмоида) и другая функция потерь (например, среднеквадратичная), формула дельты была бы иной (включала бы производную активации). Для скрытых слоёв общий принцип сохраняется всегда.
 *
 * 1.2 Вычисляем градиент функции потерь для текущего слоя = дельта-ошибки * выход-предыдущего-слоя.
 * Минимальное объяснение: Функция потерь зависит от выхода текущего слоя, который зависит от взвешенного входа (Z) текущего слоя, который зависит от весов связей предыдущего слоя с текущим и выхода предыдущего слоя
 * Конкретнее вычисляем градиент по каждому весу, соединяющему нейрон i предыдущего слоя с нейроном j выходного слоя
 * Градиент = дельта-ошибки-нейрона-j * выход-нейрона-i
 * 1.3 Корректируем веса связей выходного слоя, двигаясь в сторону, противоположную градиенту (т.к. градиент это возрастание фун ошибки, а нам надо её уменьшить, то есть спускаться), с заданной скоростью обучения
 * Конкретнее, для каждой входной связи из нейрона i предыдущего слоя в текущий нейрон выходного слоя j изменяем вес связи как
 * Wij(новый) = Wij(старый) -коэф-скорости-обучения * градиент-функции-потерь
 * При использовании метода момента при изменение веса связи дополнительно учитывается предыдущее изменение её веса (имитация инерции)
 *
 * 2. Для промежуточных (скрытых) слоёв алгоритм аналогичен, но дельта ошибки вычисляется иначе, поскольку для нейронов скрытого слоя неизвестно целевое значение.
 * Дельта i-го нейрона скрытого слоя l (δ_i^(l)) вычисляется рекуррентно через дельты нейронов следующего слоя (l+1)
 * δ_i^(l) = ( ∑_k δ_k^(l+1) · w_ik^(l+1) ) · f'(z_i^(l)), где w_ik^(l+1) — веса, связывающие данный i-й нейрон с нейронами k следующего слоя, а f'(z_i^(l)) — производная активационной функции текущего нейрона. Суммирование ведётся по всем нейронам следующего слоя, в которые передаёт сигнал текущий нейрон.
 * Градиент ошибки по весам скрытого слоя l вычисляется как произведение дельты текущего нейрона на выход соответствующего нейрона из предыдущего слоя (l-1)
 *
 * 3. Процесс повторяется для всех слоёв в обратном порядке (начиная с выходного и заканчивая первым скрытым) — это и есть обратное распространение ошибки. После обновления всех весов выполняется следующая итерация (прямой проход с новыми весами) на том же или новом обучающем примере (в зависимости от режима обучения — стохастический, пакетный или мини-батч). Итерации продолжаются до достижения критерия остановки (минимум ошибки, заданное число эпох и т.д.).
 * */
public class Net1 implements Serializable {
    /**Функции инициализации весов*/

    private static Supplier<Double> INIT_NORMAL(double std){
        return () -> {
            // Генерация случайного числа из нормального распределения
            double u1 = Math.random();
            double u2 = Math.random();
            double z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            return z * std;
        };
    }

    /**Функции*/
    public static Supplier<Double> SIMPLE_INIT = (Supplier<Double> & Serializable) () -> Math.random() - 0.5;

    //Когда использовать сигмоиду: небольшие сети. Когда идёт работа с отрицательными вероятностями
    public static UnaryOperator<Double> SIGMOID = (UnaryOperator<Double> & Serializable)x -> 1.0 / (1 + Math.exp(-x));
    public static Function<Neiron, Double> SIGMOID_DERIVATIVE =  ( Function<Neiron, Double> & Serializable) n -> (
            //SIGMOID.apply(n.iValue) * (1 - SIGMOID.apply(n.iValue)) /// классический вариант
            n.oValueRaw * (1 - n.oValueRaw) // ускоренный вариант
    );
    public static Supplier<Double> SIGMOID_INIT_XAVIER (int fanIn, int fanOut) {
        double std = Math.sqrt(2.0 / (fanIn + fanOut));
        return INIT_NORMAL(std);
    }

    //тангес
    public static UnaryOperator<Double> TANH = (UnaryOperator<Double> & Serializable) x -> Math.tanh(x);
    public static Function<Neiron, Double> TANH_DERIVATIVE = (Function<Neiron, Double> & Serializable) n -> 1.0 - n.oValueRaw * n.oValueRaw; // производная tanh: 1 - tanh²(x)
    public static Supplier<Double> TANH_INIT_XAVIER(int fanIn, int fanOut) {
        double std = 2.0/(fanIn + fanOut);// Инициализация Xavier/Glorot для tanh
        return INIT_NORMAL(std);
    }

    //Когда использовать релу: критична скорость обучения. Нельзя исп с отриц весами (теряется инф)
    public static UnaryOperator<Double> RELU = (UnaryOperator<Double> & Serializable) x -> x > 0 ? x : 0.01 * x;
    public static Function<Neiron, Double> RELU_DERIVATIVE = (Function<Neiron, Double> & Serializable) n -> n.iValue > 0 ? 1.0 : 0.01;
    public static Supplier<Double> RELU_INIT_HE(int fanIn) {
        double std = Math.sqrt(2.0 / fanIn);
        return INIT_NORMAL(std);
    }


    /**Интерфейс для тестирования нейросети*/
    public interface Estimation {
        /**
         * Проверка результата вычислений во время тестирования
         *
         * @param expectedVec     - вектор ожидаемых значений
         * @param outputVec       - вектор выходных значений
         * @param acceptableError - допустимая ошибка при которой ответ считается верным
         * @return - правильный ответ или нет
         */
        boolean process(double[] outputVec, double[] expectedVec, double acceptableError);
    }

    //Данные для Net
    @Getter
    List<Layer> layers = new ArrayList<>();


    /**Конструктор нейросети
     * @param layerCreateFuns - список функций создания слоя с получением в качестве аргумента предыдущего слоя
     * @return - нейросеть*/
    public Net1(List<Function<Layer,Layer>> layerCreateFuns){
        layers.add(layerCreateFuns.getFirst().apply(null));
        for(int i=1; i<layerCreateFuns.size(); i++)
            layers.add(layerCreateFuns.get(i).apply(layers.getLast()));

        if (!LayerInput.class.isAssignableFrom(layers.getFirst().getClass()))
            throw new RuntimeException("Первый слой должен быть входным!");
        for(int i=1; i<layerCreateFuns.size()-1; i++)
            if (!LayerMedium.class.isAssignableFrom(layers.get(i).getClass()))
                throw new RuntimeException("Между входным и выходным-и слоями должны быть только промежуточные слои");
       if (!LayerOutput.class.isAssignableFrom(layers.getLast().getClass()))
            throw new RuntimeException("Последний слой должен быть выходным!");
    }

    /**Прямое распространение сигнала (вычисление)
     * @param inputs - вектор входных значений
     * @param trainingMode - если истина, то режим обучения, иначе режим работы (влияет на дропауты)
     * @return -вектор выходных значений*/
    public double[] forward(double[] inputs, boolean trainingMode){
        ((LayerInput)layers.getFirst()).forward(inputs, trainingMode);

        for (int i=1; i<layers.size()-1; i++)
            ((LayerMedium)layers.get(i)).forward(trainingMode);

        return ((LayerOutput)layers.getLast()).forward();
    }
    public double[] forward(double[] inputs){
        return forward(inputs, false);
    }

    /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения
     * @param targets - вектор целевых значений выходных нейронов
     * @param param - гиперпараметры оптимизатора
     * @param optimizator - сам оптимизатор
     * @param batchCurrentSize - остаток от текущей пачки.
     * @param batchMaxSize - максимальный размер пачки
     * @param isEnd - если истина то это последний прогон в эпохе
     * @return - текущее значение batchCurrentSize
     * */
    private int backward(double[] targets, ParamOptimizator param, Optimizator optimizator, int batchCurrentSize, int batchMaxSize, boolean isEnd){
        /*
        Ошибка вычисляется на выходном слое.
        Затем она распространяется назад: от выходного слоя к входному.
        Каждый нейрон (кроме входных) имеет дельту (δ), которая показывает, насколько он виноват в ошибке.

        LayerOutput.backward:
            - вычисляет дельту для каждого нейрона выходного слоя.
            - обновляет веса входящих связей (iLinks) этих нейронов.
        LayerMedium.backward:
            - вычисляет дельту для каждого нейрона скрытого слоя (используя дельты следующего слоя и веса исходящих связей).
            - обновляет веса входящих связей (iLinks) этих нейронов.
        LayerInput.backward:
            - Входной слой не имеет входящих связей, а его исходящие связи (oLinks) уже обновлены на предыдущем шаге. Ничего не делаем
        */

        if (batchCurrentSize == batchMaxSize)
            isEnd = true;

        ((LayerOutput)layers.getLast()).backward(targets, optimizator, param.stepCount, batchMaxSize == -1 ? -1 : batchCurrentSize, isEnd);

        for (int i=layers.size()-2; i>0; i--)
            ((LayerMedium)layers.get(i)).backward(optimizator, batchMaxSize == -1 ? -1 : batchCurrentSize, isEnd);

        // Ничего не делаем так как исходящие из входного слоя связи уже обновлены первым промежуточным слоем, а дельты вычислять не надо так как входящих связей нет
        // ((LayerInput)layers.getFirst()).backward();

        if (isEnd) {
            param.nextStep();//следующий шаг отпимизатора только после обновления весов
            return 0;
        }
        else return batchCurrentSize+1;
    }

    /**Тренировка на наборе данных
     * @param dto - дто с даным для запуска обучения
     * @param testDto -тестовое дто для запуска тестов
     * @param testValueStop - значение при превышении которого останавливаем обучение*/
    public void trains(TrainDto dto, TestDto testDto, double testValueStop){
        //Получим оптимизатор
        ParamOptimizator param = dto.paramOptimizatorSupplier.get();
        Optimizator optimizator = param.createOptimizator();
        //Подготовимся к обучению. Расставим оптимизаторы
        for(Layer layer : layers){
            for(Neiron neiron : layer.neirons){
                neiron.dataOptimizator = optimizator.createData();
                neiron.gradAccumBias = 0.0;
                if ( neiron.iLinks != null)
                    for(Link link : neiron.iLinks) {
                        link.dataOptimizator = optimizator.createData();
                        link.gradAccumWeight = 0.0;
                    }
            }
        }
        param.reset();
        int batchSizeConst = dto.batchSizeConst;

        int epoch=0; //текущая эпоха
        int countCalcTestTmp = dto.countCalcTest;
        double testValue=0F;

        Instant start = Instant.now();
        while (epoch++<dto.maxEpochCount && testValue < testValueStop){
            Instant startEpoch = Instant.now();
            int batchCurrentSize = 0;
            for (int i = 0; i < dto.dataSize; i++) {
                if (i > 0 && i % 5000 == 0) System.out.println("processed " + i + " from " + dto.dataSize);

                //единичная тренировка
                forward(dto.fInput.apply(i), true);
                batchCurrentSize = backward(dto.fTarget.apply(i), param, optimizator, batchCurrentSize, batchSizeConst, batchSizeConst == -1 || i == dto.dataSize - 1);

                //countCalcTestTmp
                if (countCalcTestTmp>=0) {
                    if (countCalcTestTmp-- == 0) {
                        testValue = tests(testDto);
                        if (testValue>=testValueStop)
                            break;
                        System.out.println("tests = " + String.format("%.3f", testValue));
                        countCalcTestTmp = dto.countCalcTest;
                    }
                }
            }

            System.out.println("epoch=" + epoch + " duration(ms)=" + Duration.between(startEpoch, Instant.now()).toMillis());
        }

        System.out.println("all duration(ms)=" + Duration.between(start, Instant.now()).toMillis());

        if (testValue<testValueStop)
            testValue = tests(testDto);
        System.out.println("end tests = " + String.format("%.3f", testValue));
    }
    /**ДТО для запуска тренировки сети
     * @param fInput - функция получения входного вектора по номеру
     * @param fTarget - функция получения целевого вектора по номеру
     * @param dataSize - длина набора данных
     * @param maxEpochCount - максимальное количество эпох обучения
     * @param countCalcTest - после проведения какого кол-ва подходов будет запускаться тестирование для получениия оценки (-1 чтобы не запускать вообще)
     * @param paramOptimizatorSupplier - функция создающая гиперпараметры для оптимизатора
     * @param batchSizeConst - размер пачки. Использовать -1 если работаем без пачки
     */
    public static record TrainDto(Function<Integer, double[]> fInput, Function<Integer, double[]> fTarget, int dataSize, int maxEpochCount, int countCalcTest, Supplier<ParamOptimizator> paramOptimizatorSupplier, int batchSizeConst){}

    /**Запуск тестов
     * @param dto - дто для запуска тестов
     * @return - процент успешно пройдённых тестов*/
    public double tests(TestDto dto){
        int success=0;
        for(int i=0; i< dto.dataSize; i++){
            double[] actual = forward(dto.fInput.apply(i), false);
            if (dto.estimation.process(actual,dto.fTarget.apply(i), dto.acceptableError))
                success++;
        }

        return (double) success / dto.dataSize;
    }
    /**ДТО для запуска тестов
     *      * @param fInput - функция получения входного вектора по номеру
     *      * @param fTarget - функция получения целевого вектора по номеру
     *      * @param dataSize - длина набора данных
     *      * @param estimation - функция вычисляющая можно ли считать данный ответ правильным или нет
     *      * @param acceptableError - допустимая ошибка при которой ответ считается правильным*/
    public static record TestDto(Function<Integer, double[]> fInput, Function<Integer, double[]> fTarget, int dataSize, Estimation estimation, double acceptableError){ }

    /**
     * Сохранить обученную нейросеть в файл по его пути
     *
     * @param filePath - путь к файлу в который сохраняем
     * @param object   - сохраняемый объект
     */
    public static void save(String filePath, Net1 object) {
        try {
            var fileOutput = new FileOutputStream(filePath);
            var objectOutput = new ObjectOutputStream(fileOutput);
            objectOutput.writeObject(object);
            fileOutput.flush();
            objectOutput.flush();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        System.out.println("saving in " + filePath);
    }

    /**
     * Загрузить обученную нейросеть из файла
     *
     * @param filePath - путь к файлу из которого загружаем нейронку
     * @return - нейросеть
     */
    public static Net1 load(String filePath) {
        try {
            var fileInput = new FileInputStream(filePath);
            var objectInput = new ObjectInputStream(fileInput);
            Object object = objectInput.readObject();
            fileInput.close();
            objectInput.close();
            System.out.println("loading from " + filePath);
            return (Net1)object;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**Распечатать нейросеть в консоле*/
    public void  print(){
        for(int i=0; i<layers.size(); i++)
            layers.get(i).print(i);
    }

    /**вернёт индекс максимального элемента из вектора*/
    public static Integer findMax(double[] arr) {
        double max = -999F;
        Integer imax = -1;
        for (int i = 0; i < arr.length; i++)
            if (arr[i] > max) {
                max = arr[i];
                imax = i;
            }
        return imax;
    }

    /**По номеру слова создаёт массив входных данных */
    public static double[] generateVec(int number, int size, double min, double max){
        double[] voc = new double[size];
        Arrays.fill(voc, min);
        voc[number] = max;
        return voc;
    }

    /**Функция эстимации (можно ли считать данный ответ нейросети правильным). По максимальному значению*/
    public static Net1.Estimation estimationMax = (double[] outputVec, double[] expectedVec, double acceptableError) ->{
        return findMax(outputVec).equals(findMax(expectedVec));
    };

    /**Функция эстимации (можно ли считать данный ответ нейросети правильным). Функция потерь*/
    public static Net1.Estimation estimationLoss = (double[] outputVec, double[] expectedVec, double acceptableError) ->{
        double loss=0;
        for(int i=0; i<outputVec.length;i++)
            loss = loss + Math.abs(expectedVec[i]-outputVec[i]);
        return loss < acceptableError;
    };

    //------------------Оптимизаторы
    //Оптимизаторы
    @RequiredArgsConstructor
    public abstract static class Optimizator implements Serializable{
        public final ParamOptimizator param;

        /**Метод обновления веса
         * @param weight - обновляемое значение с объекта (связи или биаса для нейрона)
         * @param getDataForOptimizatorInterface - метод получения данных для оптимизации сохраняемых на объекте
         * @param grad - градиент функции потерь
         * @return изменённое значение веса*/
        public abstract double update(double weight, GetDataForOptimizatorInterface getDataForOptimizatorInterface, double grad);

        /**Создадим сохраняемые на объекте данные для оптимизатора*/
        public abstract DataOptimizator createData();
    }
    //Простой оптимизатор
    public static class OptimizatorConst extends Optimizator{
        public OptimizatorConst(ParamOptimizator param) {super(param);}

        @Override
        public double update(double weight, GetDataForOptimizatorInterface getDataForOptimizatorInterface, double grad) {
            ConstParamOptimizator param = (ConstParamOptimizator)this.param;
            return weight - param.lr *grad;
        }

        @Override
        public DataOptimizator createData() {return null;}
    }
    //ADAM-оптимизатора
    public static class OptimizatorAdam extends Optimizator{
        public OptimizatorAdam(ParamOptimizator param) {super(param);}

        @Override
        public double update(double weight, GetDataForOptimizatorInterface getDataForOptimizatorInterface, double grad) {
            AdamParamOptimizator param = (AdamParamOptimizator)this.param;
            AdamDataOptimizator adam = (AdamDataOptimizator) getDataForOptimizatorInterface.getDataOptimizator();

            // Обновление моментов Adam
            adam.m = param.beta1 * adam.m + (1 - param.beta1) * grad;
            adam.v = param.beta2 * adam.v + (1 - param.beta2) * grad * grad;
            // Смещение моментов (bias correction)
            double mHat = adam.m * param.invCorrection1;
            double vHat = adam.v * param.invCorrection2 ;
            // Обновление веса
            return weight - param.lr * mHat / (Math.sqrt(vHat) + param.epsilon);
        }

        @Override
        public DataOptimizator createData() {return new AdamDataOptimizator();}
    }
    //дата для оптимизатора
    public static abstract class DataOptimizator implements Serializable{};
    public static class AdamDataOptimizator extends DataOptimizator{
        public double m = 0.0;  // первый момент
        public double v = 0.0;  // второй момент
    }
    //Интерфейс для получения даты для оптимизатора
    public interface GetDataForOptimizatorInterface{
        DataOptimizator getDataOptimizator();
    }
    //гиперпараметры для оптимизатора
    public static abstract class ParamOptimizator implements Serializable{
        public int stepCount = 1; // счётчик шагов (обновляется при каждом изменении весов в backward)

        public abstract Optimizator createOptimizator();
        public void nextStep(){stepCount++;}
        public  void reset(){stepCount = 1;};
    };
    @AllArgsConstructor
    public static class ConstParamOptimizator extends ParamOptimizator{
        public final double lr; //коэф обучения

        @Override
        public Optimizator createOptimizator() {return new OptimizatorConst(this);}
    }
    @NoArgsConstructor
    @Accessors(chain = true)
    public static class AdamParamOptimizator extends ParamOptimizator{
        @Setter public double lr = 0.001; //коэф обучения
        /**Параметры  оптимизатора*/
        @Setter public double beta1 = 0.9;
        @Setter public double beta2 = 0.999;
        @Setter public double epsilon = 1e-8;
        /**Параметры для ускорения расчёта*/
        double beta1Pow;
        double beta2Pow;
        double invCorrection1;
        double invCorrection2;

        @Override
        public Optimizator createOptimizator() {return new OptimizatorAdam(this);}
        @Override
        public void nextStep(){
            super.nextStep();
            beta1Pow *= beta1;
            beta2Pow *= beta2;
            invCorrection1 = 1.0 / (1.0 - beta1Pow);
            invCorrection2 = 1.0 / (1.0 - beta2Pow);
        }
        public  void reset(){
            super.reset();
            beta1Pow = beta1;
            beta2Pow = beta2;
            invCorrection1 = 1.0 / (1.0 - beta1Pow);
            invCorrection2 = 1.0 / (1.0 - beta2Pow);
        };
    }

    //------------------LINK
    /**Связь между нейронами*/
    @FieldDefaults(level = AccessLevel.PUBLIC)
    public static class Link implements Serializable, GetDataForOptimizatorInterface{
        double weight = 0F; //вес связи
        public double gradAccumWeight = 0.0;      // сумма градиентов за батч

        Neiron iNeiron;
        Neiron oNeiron;

        DataOptimizator dataOptimizator;
        @Override
        public DataOptimizator getDataOptimizator() {
            return dataOptimizator;
        }

        /**Создать связь между нейронами
         * @param iNeiron - входной нейрон
         * @param oNeiron - выходной нейрон
         * @param initWeightFun - функция инициализации весов
         * @return - созданная связь*/
        public static Link createLink(Neiron iNeiron, Neiron oNeiron, Supplier<Double> initWeightFun){
            Link link = new Link();
            link.weight = initWeightFun.get();
            link.iNeiron = iNeiron;
            link.oNeiron = oNeiron;

            iNeiron.oLinks.add(link);
            oNeiron.iLinks.add(link);
            return link;
        }

        public void print(int layerNumber, int neironNumber, int linkNumber){
            System.out.println("\t\t\tlink "+layerNumber+"_"+neironNumber+" => "+(layerNumber+1)+"_"+linkNumber+" : "+String.format("%.4f", weight));
        }
    }
    
    //------------------NEIRON
    /**Нейрон*/
    @FieldDefaults(level = AccessLevel.PUBLIC)
    public static class Neiron implements Serializable, GetDataForOptimizatorInterface{
        double dropoutMask = 1.0; // маска dropout (0 - отключен, 1 - активен)

        double bias = Math.random() * 0.1 - 0.05; //смещение
        public double gradAccumBias = 0.0; // сумма градиентов для смещения

        double iValue=0F;//входное значение (взвешенная сумма переданных сигналов от нейронов предыдущего слоя) + смещение
        double oValue;//выходное значение - то что передаётся от этого нейрона нейрону следующего слоя.
        double oValueRaw;//сырое выходное значение (без изменений в связи с дроп-аутом) только для промежуточных слоёв
        double delta; //дельта ошибки
        List<Link> iLinks = new ArrayList<>();
        List<Link> oLinks = new ArrayList<>();

        DataOptimizator dataOptimizator;
        @Override
        public DataOptimizator getDataOptimizator() {
            return dataOptimizator;
        }

        /**Создать нейрон входного слоя*/
        public static Neiron createInputNeiron(){Neiron n = new Neiron();n.iLinks=null;return n;}
        /**Создать нейрон промежуточного слоя*/
        public static Neiron createMediumNeiron(){return new Neiron();}
        /**Создать нейрон выходного слоя*/
        public static Neiron createOutputNeiron(){Neiron n = new Neiron();n.oLinks=null;return n;}

        public void print(int layerNumber, int neironNumber){
            System.out.println("\t\tneiron "+layerNumber+"_"+neironNumber+" ("+"b="+String.format("%.4f", bias)+", delta="+String.format("%.4f",delta)+", iValue="+String.format("%.4f",iValue)+", oValue="+String.format("%.4f",oValue)+")");
            if(oLinks==null) return;
            for(int i=0; i<oLinks.size(); i++)
                oLinks.get(i).print(layerNumber, neironNumber, i);
        }
    }
    
    //------------------LAYER
    /**Абстрактный класс слоя*/
    public abstract static class Layer implements Serializable{
        final public List<Neiron> neirons = new ArrayList<>();

        /**Вывести на печать*/
        public void print(int layerNumber){
            System.out.println("\tLayer number "+layerNumber);
            for(int i=0; i<neirons.size(); i++)
                neirons.get(i).print(layerNumber, i);

        }
    }

    /**Входной слой*/
    public static class LayerInput extends Layer {
        double dropoutRate = 0.0; // процент дропаута (0.0 - нет дропаута, 0.5 - 50%)

        /**Конструктор входного слоя
         * @param nCount - количество нейронов в входном слою
         * @param dropoutRate - процент дропаута (от 0.0 до 0.99)
         * @return - входной слой*/
        public LayerInput(int nCount, double dropoutRate){
            this.dropoutRate = dropoutRate;

            for(int i =0; i<nCount; i++)
                this.neirons.add(Neiron.createInputNeiron());
        }

        /**Прямой проход для нейронов слоя (вычисление)
         * @param inputs - вектор входных данных
         * @param trainingMode - если истина, то режим обучения, иначе режим работы*/
        public void forward(double[] inputs, boolean trainingMode){
            assert (inputs.length!=neirons.size()) : "Несовпадение размерности";

            for(int i=0; i<inputs.length; i++) {
                neirons.get(i).oValue = inputs[i];
                neirons.get(i).oValueRaw = neirons.get(i).oValue;
                if (trainingMode && dropoutRate>0)
                    if (Math.random() < dropoutRate)
                        neirons.get(i).oValue = 0.0;
            }
        }

        /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения*/
        public void backward(){
            // Ничего не делаем так как исходящие из входного слоя связи уже обновлены первым промежуточным слоем, а дельты вычислять не надо так как входящих связей нет
        }
    }

    /**Промежуточный слой*/
    @FieldDefaults(level = AccessLevel.PROTECTED)
    public static class LayerMedium extends Layer {
        UnaryOperator<Double> activation; //функция активации
        Function<Neiron, Double> activationDer; //производная функции активации

        private double dropoutRate = 0.0; // процент дропаута (0.0 - нет дропаута, 0.5 - 50%)

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


                n.iValue=n.bias;
                for (Link link : n.iLinks)
                    n.iValue  += link.iNeiron.oValue * link.weight;
                n.oValue=this.activation.apply(n.iValue);

                n.oValueRaw = n.oValue;
                // Для inverted dropout: масштабируем только при обучении
                if (trainingMode && dropoutRate > 0.0 && n.dropoutMask == 1.0) {
                    n.oValue *= 1.0 / (1.0 - dropoutRate); // inverted dropout
                }
            }
        }

        /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения
         * @param optimizator - оптимизатор
         * @param batchSize - размер пачки. Если -1 то без пакетного режима
         * @param weightCorrectFlg - флаг того надо ли корректировать веса
         * */
        public void backward(Optimizator optimizator, int batchSize, boolean weightCorrectFlg){
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
                    delta += outLink.oNeiron.delta * outLink.weight;
                neiron.delta = activationDer.apply(neiron) * delta;

                if (batchSize == -1)
                    neiron.bias = optimizator.update(neiron.bias, neiron, neiron.delta);//производная по bias равна delta (так как bias влияет напрямую на iValue)
                else
                    if (weightCorrectFlg){
                        neiron.bias = optimizator.update(neiron.bias, neiron, neiron.gradAccumBias / batchSize);
                        neiron.gradAccumBias = 0.0;
                    }
                    else
                        neiron.gradAccumBias += neiron.delta;

                //корректируем веса входящих связей для нейрона
                for (Link inLink : neiron.iLinks) {
                    double grad = inLink.iNeiron.oValue * neiron.delta;

                    if (batchSize == -1)
                        inLink.weight = optimizator.update(inLink.weight, inLink, grad);//inLink.weight -= lr * grad;
                    else
                    if (weightCorrectFlg){
                        inLink.weight = optimizator.update(inLink.weight, inLink, inLink.gradAccumWeight / batchSize);
                        inLink.gradAccumWeight = 0.0;
                    }
                    else
                        inLink.gradAccumWeight += grad;
                }
            }
        }

        /**Промежуточный слой - сигмоида*/
        public static class LayerMediumSigmoid extends LayerMedium {
            public LayerMediumSigmoid(Layer prevoisLayer, int nCount, double dropoutRate){
                super(nCount, prevoisLayer, Net1.SIGMOID, Net1.SIGMOID_DERIVATIVE, Net1.SIGMOID_INIT_XAVIER(prevoisLayer.neirons.size(), nCount), dropoutRate);
            }
        }
        /**Промежуточный слой - тангес*/
        public static class LayerMediumTanh extends LayerMedium {
            public LayerMediumTanh(Layer prevoisLayer, int nCount, double dropoutRate){
                super(nCount, prevoisLayer, Net1.TANH, Net1.TANH_DERIVATIVE, Net1.TANH_INIT_XAVIER(prevoisLayer.neirons.size(), nCount), dropoutRate);
            }
        }
        /**Промежуточный слой - релу*/
        public static class LayerMediumRelu extends LayerMedium {
            public LayerMediumRelu(Layer prevoisLayer, int nCount, double dropoutRate){
                super(nCount, prevoisLayer, Net1.RELU, Net1.RELU_DERIVATIVE, Net1.RELU_INIT_HE(prevoisLayer.neirons.size()), dropoutRate);
            }
        }
    }

    /**Выходной слой*/
    public static class LayerOutput extends Layer {
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

                n.iValue=n.bias;
                for (Link link : n.iLinks)
                    n.iValue  += link.iNeiron.oValue * link.weight;
                n.oValue=this.activation.apply(n.iValue);
                n.oValueRaw = n.oValue;
                outputs[i]=n.oValue;
            }

            return outputs;
        }

        /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения
         * @param targets - вектор целевых значений выходных нейронов
         * @param batchSize - размер пачки. Если -1 то без пакетного режима
         * @param weightCorrectFlg - флаг того надо ли корректировать веса
         * @param optimizator - оптимизатор*/
        public void backward(double[] targets, Optimizator optimizator, int stepCount, int batchSize, boolean weightCorrectFlg){
            assert targets.length!=neirons.size() : "Несовпадение размерности";

            //Вычислим дельту ошибки нейронов выходного слоя
            for(int i=0; i<targets.length; i++){
                Neiron neiron = neirons.get(i);

                //дельта ошибки на выходе нейрона = производная функции потерь * производная функции активации
                calcDelta(neiron, targets[i]);

                //вычисляем изменение смещения
                if (batchSize == -1)
                    neiron.bias=optimizator.update(neiron.bias, neiron, neiron.delta);//neiron.bias -=neiron.delta*lr;
                else
                    if (weightCorrectFlg){
                        neiron.bias = optimizator.update(neiron.bias, neiron, neiron.gradAccumBias / batchSize);
                        neiron.gradAccumBias = 0.0;
                    }
                    else
                        neiron.gradAccumBias += neiron.delta;


                //корректируем веса входящих связей для нейрона
                for (Link inLink : neiron.iLinks) {
                    double grad = inLink.iNeiron.oValue * neiron.delta;

                    if (batchSize == -1)
                        inLink.weight=optimizator.update(inLink.weight, inLink, grad);//inLink.weight -=lr * grad;
                    else
                    if (weightCorrectFlg){
                        inLink.weight = optimizator.update(inLink.weight, inLink, inLink.gradAccumWeight / batchSize);
                        inLink.gradAccumWeight = 0.0;
                    }
                    else
                        inLink.gradAccumWeight += grad;
                }
            }
        }

        /**Вычисление дельты выходного нейрона*/
        protected void calcDelta(Neiron neiron, double targets) {
            //дельта ошибки на выходе нейрона = производная функции потерь * производная функции активации
            //для  MSE  loss = (target - output)^2 производная по output: 2*(output - target) (но обычно берут (output - target)
            neiron.delta=(neiron.oValue - targets) * activationDer.apply(neiron);
        }

        /**Выходной слой - сигмоида*/
        public static class LayerOutputSigmoid extends LayerOutput {
            public LayerOutputSigmoid(Layer prevoisLayer, int nCount){
                super(nCount, prevoisLayer, Net1.SIGMOID, Net1.SIGMOID_DERIVATIVE, Net1.SIGMOID_INIT_XAVIER(prevoisLayer.neirons.size(), nCount));
            }
        }
        /**Выходной слой - тангес*/
        public static class LayerOutputTanh extends LayerOutput {
            public LayerOutputTanh(Layer prevoisLayer, int nCount){
                super(nCount, prevoisLayer, Net1.TANH, Net1.TANH_DERIVATIVE, Net1.TANH_INIT_XAVIER(prevoisLayer.neirons.size(), nCount));
            }
        }
        /**Выходной слой - релу*/
        public static class LayerOutputRelu extends LayerOutput {
            public LayerOutputRelu(Layer prevoisLayer, int nCount){
                super(nCount, prevoisLayer, Net1.RELU, Net1.RELU_DERIVATIVE, Net1.RELU_INIT_HE(prevoisLayer.neirons.size()));
            }
        }

        /**
         * Выходной слой с softmax и кросс-энтропией.
         * Предполагается, что целевой вектор имеет one-hot кодирование (то есть там только одна единичка, остальное 0)
         * Для софтмакса коэффициент обучения обычно нужен меньше чем для сигмоида и тангеса примерно раз в десять.
         */
        public static class LayerOutputSoftmaxAndCrossEntity extends LayerOutput {

            /**
             * Конструктор выходного softmax-слоя.
             * @param previousLayer предыдущий слой
             * @param nCount количество нейронов в слое (число классов)
             */
            public LayerOutputSoftmaxAndCrossEntity(Layer previousLayer, int nCount) {
                // Передаём фиктивные функции активации и производной,
                // они не будут использоваться, так как forward и calcDelta переопределены.
                // Инициализация весов — Xavier (подходит для softmax).
                super(nCount, previousLayer,
                        x -> x,                 // фиктивная активация
                        n -> 1.0,               // фиктивная производная
                        SIGMOID_INIT_XAVIER(previousLayer.neirons.size(), nCount) // Xavier
                );
            }

            @Override
            public double[] forward() {
                int n = neirons.size();
                double[] iValues = new double[n];

                // 1. Вычисляем взвешенные суммы (Z = W·X + b) для всех нейронов
                for (int i = 0; i < n; i++) {
                    Neiron neiron = neirons.get(i);
                    double sum = neiron.bias;
                    for (Link link : neiron.iLinks) {
                        sum += link.iNeiron.oValue * link.weight;
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

            @Override
            protected void calcDelta(Neiron neiron, double target) {
                // Для комбинации softmax + кросс-энтропия дельта равна (output - target) но только если целевой вектор это one-hot вектор
                neiron.delta = neiron.oValue - target;
            }
        }
    }
}
