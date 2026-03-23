package neiro.simple.mlp.stabile.task;

import lombok.SneakyThrows;
import neiro.simple.mlp.stabile.dto.Examples;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**Задача распознавания рукописных шрифтов mnist*/
public class TaskMNIST implements ITask {
    public Examples getData(String path) {
        //MNIST
        List<double[]> datasTrain = new ArrayList<>();
        List<double[]> targetsTrain = new ArrayList<>();
        List<double[]> datasTest = new ArrayList<>();
        List<double[]> targetsTest = new ArrayList<>();

        prepareDataForMnist(targetsTrain, datasTrain, "d:\\Work\\Project\\0files\\mnist\\mnist_train.csv");
        prepareDataForMnist(targetsTest, datasTest, "d:\\Work\\Project\\0files\\mnist\\mnist_test.csv");

        return new Examples(
                convertToExampleArray(datasTrain, targetsTrain),
                convertToExampleArray(datasTest, targetsTest)
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
