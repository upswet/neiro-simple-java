package neiro.simple.mlp;

import lombok.SneakyThrows;

import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import static neiro.simple.mlp.Net.estimationLoss;
import static neiro.simple.mlp.Net.estimationMax;

/**
 * Тестируем нашу mlp (полносвязанную) сеть
 */
public class ExampleMLP {
    /**
     * Задача исключительного или
     */
    public static void xor() {
        //XOR
        Net net = new Net(List.of(
                (layer) -> new Net.LayerInput(2, 0.0),
                (layer) -> new Net.LayerMedium.LayerMediumTanh(layer, 3, 0.0),
                (layer) -> new Net.LayerOutput.LayerOutputTanh(layer,1)
        ));

        /*Net net = new Net(List.of(
                (layer) -> new Net.LayerInput(2, 0.0),
                (layer) -> new Net.LayerMedium.LayerMediumRelu(layer, 3, 0.0),
                (layer) -> new Net.LayerOutput.LayerOutputRelu(layer, 1)
        ));*/


        Net.save("net1.save", net);
        net = Net.load("net1.save");
        net.print();

        double[][] inputs = new double[][]{
                new double[]{0F, 0F},
                new double[]{1F, 0F},
                new double[]{0F, 1F},
                new double[]{1F, 1F},
        };
        double[][] targets = new double[][]{
                new double[]{0F},
                new double[]{1F},
                new double[]{1F},
                new double[]{0F},
        };

        net.trains(
                new Net.TrainDto(
                        (Integer i) -> inputs[i],
                        (Integer i) -> targets[i],
                        inputs.length,
                        5000,
                        100,
                        (Supplier<Net.ParamOptimizator> & Serializable) () -> new Net.ConstParamOptimizator(0.2),
                        //(Supplier<Net.ParamOptimizator> & Serializable) () -> (new Net.AdamParamOptimizator()).setLr(0.001),
                        -1
                ),
                new Net.TestDto(
                        (Integer i) -> inputs[i],
                        (Integer i) -> targets[i],
                        inputs.length,
                        estimationLoss,
                        0.1
                ),
                0.98
        );


        System.out.println("{1,1} = " + Arrays.toString(net.forward((new double[]{1, 1}))));
        System.out.println("{1,0} = " + Arrays.toString(net.forward((new double[]{1, 0}))));
        System.out.println("{0,1} = " + Arrays.toString(net.forward((new double[]{0, 1}))));
        System.out.println("{0,0} = " + Arrays.toString(net.forward((new double[]{0, 0}))));
    }


    /**
     * Задача распозновазния рукописных шрифтов mnist
     */
    public static void mnist() {
        //MNIST
        List<double[]> datasTrain = new ArrayList<>();
        List<double[]> targetsTrain = new ArrayList<>();
        List<double[]> datasTest = new ArrayList<>();
        List<double[]> targetsTest = new ArrayList<>();

        prepareDataForMnist(targetsTrain, datasTrain, "d:\\Work\\Project\\0files\\mnist\\mnist_train.csv");
        prepareDataForMnist(targetsTest, datasTest, "d:\\Work\\Project\\0files\\mnist\\mnist_test.csv");

        /*Net net = new Net(List.of(
                (layer) -> new Net.LayerInput(784, 0.0),
                (layer) -> new Net.LayerMedium.LayerMediumTanh(layer, 100, 0.0),
                (layer) -> new Net.LayerOutput.LayerOutputSigmoid(layer, 10)
        ));*/
        Net net = new Net(List.of(
                (layer) -> new Net.LayerInput(784, 0.0),
                (layer) -> new Net.LayerMedium.LayerMediumTanh(layer, 100, 0.0),
                (layer) -> new Net.LayerOutput.LayerOutputSoftmaxAndCrossEntity(layer, 10)
        ));

        double[][] inputs = datasTrain.toArray(double[][]::new);
        double[][] targets = targetsTrain.toArray(double[][]::new);

        double[][] inputsTest = datasTest.toArray(double[][]::new);
        double[][] targetTest = targetsTest.toArray(double[][]::new);

        net.trains(
                new Net.TrainDto(
                        (Integer i) -> inputs[i],
                        (Integer i) -> targets[i],
                        inputs.length,
                        1,
                        -1,
                        //(Supplier<Net.ParamOptimizator> & Serializable) () -> new Net.ConstParamOptimizator(0.02),
                        (Supplier<Net.ParamOptimizator> & Serializable) () -> (new Net.AdamParamOptimizator()).setLr(0.001),
                        100
                ),
                new Net.TestDto(
                        (Integer i) -> inputsTest[i],
                        (Integer i) -> targetTest[i],
                        inputsTest.length,
                        Net.estimationMax,
                        0.01
                ),
                0.98
        );
    }

    /**
     * Подготовка данных MNIST
     */
    @SneakyThrows
    private static void prepareDataForMnist(List<double[]> targets, List<double[]> datas, String fileName) {
        List<String> list = Files.readAllLines(Paths.get(fileName));
        double MAX = 1;//0.99;
        double MIN = 0;//0.01;
        for (String s : list) {
            String[] sarr = s.split(",");

            double[] target = new double[10];
            Arrays.fill(target, MIN);
            target[Integer.valueOf(sarr[0])] = MAX;
            targets.add(target);

            double[] data = new double[sarr.length - 1];
            for (int j = 1; j < sarr.length; j++)
                data[j - 1] = (Double.valueOf(sarr[j]) / 255.0) * MAX + MIN;

            datas.add(data);
        }
    }
}
