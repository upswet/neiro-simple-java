package neiro.simple.mlp.old.ver1;

import lombok.SneakyThrows;

import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Supplier;

import static neiro.simple.mlp.old.ver1.Net1.estimationLoss;

/**
 * Тестируем нашу mlp (полносвязанную) сеть
 */
public class ExampleMLP {
    /**
     * Задача исключительного или
     */
    public static void xor() {
        //XOR
        Net1 net1 = new Net1(List.of(
                (layer) -> new Net1.LayerInput(2, 0.0),
                (layer) -> new Net1.LayerMedium.LayerMediumTanh(layer, 3, 0.0),
                (layer) -> new Net1.LayerOutput.LayerOutputTanh(layer,1)
        ));

        /*Net net = new Net(List.of(
                (layer) -> new Net.LayerInput(2, 0.0),
                (layer) -> new Net.LayerMedium.LayerMediumRelu(layer, 3, 0.0),
                (layer) -> new Net.LayerOutput.LayerOutputRelu(layer, 1)
        ));*/


        Net1.save("net1.save", net1);
        net1 = Net1.load("net1.save");
        net1.print();

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

        net1.trains(
                new Net1.TrainDto(
                        (Integer i) -> inputs[i],
                        (Integer i) -> targets[i],
                        inputs.length,
                        5000,
                        100,
                        (Supplier<Net1.ParamOptimizator> & Serializable) () -> new Net1.ConstParamOptimizator(0.2),
                        //(Supplier<Net.ParamOptimizator> & Serializable) () -> (new Net.AdamParamOptimizator()).setLr(0.001),
                        -1
                ),
                new Net1.TestDto(
                        (Integer i) -> inputs[i],
                        (Integer i) -> targets[i],
                        inputs.length,
                        estimationLoss,
                        0.1
                ),
                0.98
        );


        System.out.println("{1,1} = " + Arrays.toString(net1.forward((new double[]{1, 1}))));
        System.out.println("{1,0} = " + Arrays.toString(net1.forward((new double[]{1, 0}))));
        System.out.println("{0,1} = " + Arrays.toString(net1.forward((new double[]{0, 1}))));
        System.out.println("{0,0} = " + Arrays.toString(net1.forward((new double[]{0, 0}))));
    }


    /**
     * Задача распознавания рукописных шрифтов mnist
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
        Net1 net1 = new Net1(List.of(
                (layer) -> new Net1.LayerInput(784, 0.0),
                (layer) -> new Net1.LayerMedium.LayerMediumTanh(layer, 100, 0.0),
                (layer) -> new Net1.LayerOutput.LayerOutputSoftmaxAndCrossEntity(layer, 10)
        ));

        double[][] inputs = datasTrain.toArray(double[][]::new);
        double[][] targets = targetsTrain.toArray(double[][]::new);

        double[][] inputsTest = datasTest.toArray(double[][]::new);
        double[][] targetTest = targetsTest.toArray(double[][]::new);

        net1.trains(
                new Net1.TrainDto(
                        (Integer i) -> inputs[i],
                        (Integer i) -> targets[i],
                        inputs.length,
                        1,
                        -1,
                        (Supplier<Net1.ParamOptimizator> & Serializable) () -> new Net1.ConstParamOptimizator(0.02),
                        //(Supplier<Net1.ParamOptimizator> & Serializable) () -> (new Net1.AdamParamOptimizator()).setLr(0.001),
                        -1
                ),
                new Net1.TestDto(
                        (Integer i) -> inputsTest[i],
                        (Integer i) -> targetTest[i],
                        inputsTest.length,
                        Net1.estimationMax,
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

    /**Задача распознавания Ирис-ов*/
    public static void iris(){
        List<double[]> datasTrain = new ArrayList<>();
        List<double[]> targetsTrain = new ArrayList<>();
        List<double[]> datasTest = new ArrayList<>();
        List<double[]> targetsTest = new ArrayList<>();

        prepareDataForIris(targetsTrain, datasTrain, "d:\\Work\\Project\\0files\\iris\\iris.data.csv");
        myShuffle(targetsTrain, datasTrain);
        datasTest = datasTrain.subList(0, 10);
        datasTrain = datasTrain.subList(11,datasTrain.size());
        targetsTest = targetsTrain.subList(0,10);
        targetsTrain = targetsTrain.subList(11, targetsTrain.size());

        Net1 net1 = new Net1(List.of(
                (layer) -> new Net1.LayerInput(4, 0.0),
                (layer) -> new Net1.LayerMedium.LayerMediumTanh(layer, 10, 0.0),
                (layer) -> new Net1.LayerOutput.LayerOutputSoftmaxAndCrossEntity(layer, 3)
        ));

        double[][] inputs = datasTrain.toArray(double[][]::new);
        double[][] targets = targetsTrain.toArray(double[][]::new);

        double[][] inputsTest = datasTest.toArray(double[][]::new);
        double[][] targetTest = targetsTest.toArray(double[][]::new);

        net1.trains(
                new Net1.TrainDto(
                        (Integer i) -> inputs[i],
                        (Integer i) -> targets[i],
                        inputs.length,
                        20,
                        -1,
                        //(Supplier<Net.ParamOptimizator> & Serializable) () -> new Net.ConstParamOptimizator(0.02),
                        (Supplier<Net1.ParamOptimizator> & Serializable) () -> (new Net1.AdamParamOptimizator()).setLr(0.01),
                        10
                ),
                new Net1.TestDto(
                        (Integer i) -> inputsTest[i],
                        (Integer i) -> targetTest[i],
                        inputsTest.length,
                        Net1.estimationMax,
                        0.01
                ),
                0.98
        );
    }

    /**
     * Подготовка данных Iris
     */
    @SneakyThrows
    private static void prepareDataForIris(List<double[]> targets, List<double[]> datas, String fileName) {
        List<String> list = Files.readAllLines(Paths.get(fileName));
        double MAX = 1;
        double MIN = 0;
        for (String s : list) {
            String[] sarr = s.split(",");

            double[] target = new double[3];
            Arrays.fill(target, MIN);
            switch (sarr[sarr.length-1]){
                case "Iris-setosa" -> target[0] = MAX;
                case "Iris-versicolor" -> target[1] = MAX;
                case "Iris-virginica" -> target[2] = MAX;
            }
            targets.add(target);

            double[] data = new double[sarr.length - 1];
            for (int j = 0; j < sarr.length-1; j++)
                data[j] = (Double.valueOf(sarr[j]) / 10) * MAX + MIN;

            datas.add(data);
        }
    }

    /**Синхронно перемешать списки входных и целевых данных*/
    private static void myShuffle(List<double[]> targets, List<double[]> datas){
        record Pair(double[] left, double[] right){};

        List<Pair> list = new ArrayList<>();
        for(int i =0; i<targets.size(); i++)
            list.add(new Pair(targets.get(i), datas.get(i)));
        Collections.shuffle(list);

        targets.clear();
        datas.clear();
        for(Pair pair : list){
            targets.add(pair.left());
            datas.add(pair.right());
        }
    }
}
