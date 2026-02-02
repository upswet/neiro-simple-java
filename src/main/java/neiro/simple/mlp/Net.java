package neiro.simple.mlp;

import lombok.AllArgsConstructor;

import java.io.*;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

public class Net implements Serializable {
    private static Supplier<Double>  GENERATE_NORMAL(double std){
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
    public static Function<Layer.Neiron, Double> SIGMOID_DERIVATIVE =  ( Function<Layer.Neiron, Double> & Serializable) n -> (
            //SIGMOID.apply(n.iValue) * (1 - SIGMOID.apply(n.iValue)) /// классический вариант
            n.oValue * (1 - n.oValue) // ускоренный вариант
    );
    public static Supplier<Double> SIGMOID_INIT_XAVIER (int fanIn, int fanOut) {
        double std = Math.sqrt(2.0 / (fanIn + fanOut));
        return GENERATE_NORMAL(std);
    }

    //тангес
    public static UnaryOperator<Double> TANH = (UnaryOperator<Double> & Serializable) x -> Math.tanh(x);
    public static Function<Layer.Neiron, Double> TANH_DERIVATIVE = (Function<Layer.Neiron, Double> & Serializable) n -> 1.0 - n.oValue * n.oValue; // производная tanh: 1 - tanh²(x)
    public static Supplier<Double> TANH_INIT_XAVIER(int fanIn, int fanOut) {
        double std = 2.0/(fanIn + fanOut);// Инициализация Xavier/Glorot для tanh
        return GENERATE_NORMAL(std);
    }

    //Когда использовать релу: критична скорость обучения. Нельзя исп с отриц весами (теряется инф)
    public static UnaryOperator<Double> RELU = (UnaryOperator<Double> & Serializable) x -> x > 0 ? x : 0.01 * x;
    public static Function<Layer.Neiron, Double> RELU_DERIVATIVE = (Function<Layer.Neiron, Double> & Serializable) n -> n.iValue > 0 ? 1.0 : 0.01;
    public static Supplier<Double> RELU_INIT_HE(int fanIn) {
        double std = Math.sqrt(2.0 / fanIn);
        return GENERATE_NORMAL(std);
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

    List<Layer> layers = new ArrayList<>();

    /**Конструктор нейросети
     * @param layerCreateFuns - список функций создания слоя с получением в качестве аргумента предыдущего слоя
     * @return - нейросеть*/
    public Net(List<Function<Layer,Layer>> layerCreateFuns){
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
     * @param lr - коэффициент обучения
     * */
    private void backward(double[] targets, double lr){
        /*
        Ошибка вычисляется на выходном слое.
        Затем она распространяется назад: от выходного слоя к входному.
        Каждый нейрон (кроме входных) имеет дельту (δ), которая показывает, насколько он виноват в ошибке.
        Веса связей обновляются по правилу: Δw = -η * δ * x, где η - скорость обучения, δ - дельта нейрона, на который ведёт связь, x - выход нейрона, с которого идёт связь.

        LayerOutput.backward:
            - вычисляет дельту для каждого нейрона выходного слоя.
            - обновляет веса входящих связей (iLinks) этих нейронов.
        LayerMedium.backward:
            - вычисляет дельту для каждого нейрона скрытого слоя (используя дельты следующего слоя и веса исходящих связей).
            - обновляет веса входящих связей (iLinks) этих нейронов.
        LayerInput.backward:
            - Входной слой не имеет входящих связей, а его исходящие связи (oLinks) уже обновлены на предыдущем шаге. Ничего не делаем
        */

        ((LayerOutput)layers.getLast()).backward(targets, lr);

        for (int i=layers.size()-2; i>0; i--)
            ((LayerMedium)layers.get(i)).backward(lr, true);

        // Ничего не делаем так как исходящие из входного слоя связи уже обновлены первым промежуточным слоем, а дельты вычислять не надо так как входящих связей нет
        // ((LayerInput)layers.getFirst()).backward();
    }

    /**Тренировка на наборе данных
     * @param inputs - набор входных векторов
     * @param targets - набор целевых векторов
     * @param lr - коэффициент обучения
     * @param maxEpochCount - максимальное количество эпох обучения
     * @param countCalcTest - после проведения какого кол-ва подходов будет запускаться тестирование для получениия оценки (-1 чтобы не запускать вообще)
     * @param teatParam - параметры для теста
     * @param testValueStop - значение при превышении которого останавливаем обучение*/
    public void trains(double[][] inputs, double[][] targets, double lr, int maxEpochCount, int countCalcTest, TestWrapper teatParam, double testValueStop){
        if(inputs.length!=targets.length) throw new RuntimeException("Несовпадение размерности");

        int epoch=0; //текущая эпоха
        int countCalcTestTmp = countCalcTest;
        double testValue=0F;

        Instant start = Instant.now();
        while (epoch++<maxEpochCount && testValue < testValueStop){
            Instant startEpoch = Instant.now();
            for (int i = 0; i < targets.length; i++) {
                if (i > 0 && i % 5000 == 0) System.out.println("processed " + i + " from " + targets.length);

                //единичная тренировка
                forward(inputs[i], true);
                backward(targets[i], lr);

                //countCalcTestTmp
                if (countCalcTestTmp>=0) {
                    if (countCalcTestTmp-- == 0) {
                        testValue = tests(teatParam);
                        if (testValue>=testValueStop)
                            break;
                        System.out.println("tests = " + String.format("%.3f", testValue));
                        countCalcTestTmp = countCalcTest;
                    }
                }
            }

            System.out.println("epoch=" + epoch + " duration(ms)=" + Duration.between(startEpoch, Instant.now()).toMillis());
        }

        System.out.println("all duration(ms)=" + Duration.between(start, Instant.now()).toMillis());

        if (testValue<testValueStop)
            testValue = tests(teatParam);
        System.out.println("end tests = " + String.format("%.3f", testValue));
    }

    /**Запуск тестов
     * @param testParam - параметры запуска тестов
     * @return - процент успешно пройдённых тестов*/
    public double tests(TestWrapper testParam){
        if (testParam.inputs.length != testParam.targets.length) throw new RuntimeException("Несоответствие размерности!");

        int success=0;
        for(int i=0; i< testParam.inputs.length; i++){
            double[] actual = forward(testParam.inputs[i], false);
            if (testParam.estimation.process(actual, testParam.targets[i], testParam.acceptableError))
                success++;
        }

        return (double) success / testParam.inputs.length;
    }

    /**
     * Сохранить обученную нейросеть в файл по его пути
     *
     * @param filePath - путь к файлу в который сохраняем
     * @param object   - сохраняемый объект
     */
    public static void save(String filePath, Net object) {
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
    public static Net load(String filePath) {
        try {
            var fileInput = new FileInputStream(filePath);
            var objectInput = new ObjectInputStream(fileInput);
            Object object = objectInput.readObject();
            fileInput.close();
            objectInput.close();
            System.out.println("loading from " + filePath);
            return (Net)object;
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

    public static Net.Estimation estimationMax = (double[] outputVec, double[] expectedVec, double acceptableError) ->{
        return findMax(outputVec).equals(findMax(expectedVec));
    };

    public static Net.Estimation estimationLoss = (double[] outputVec, double[] expectedVec, double acceptableError) ->{
        double loss=0;
        for(int i=0; i<outputVec.length;i++)
            loss = loss + Math.abs(expectedVec[i]-outputVec[i]);
        return loss < acceptableError;
    };

    /**Враппер для тестовых параметров
     *      * @param inputs - набор входных векторов
     *      * @param targets - набор целевых векторов
     *      * @param estimation - функция проверки результатов тестирования
     *      * @param acceptableError - допустимая ошибка при которой ответ считается верным
     *      */
    @AllArgsConstructor
    public static class TestWrapper{
        double[][] inputs;
        double[][] targets;
        Estimation estimation;
        double acceptableError;
    }
}
