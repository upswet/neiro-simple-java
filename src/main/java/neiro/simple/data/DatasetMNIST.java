package neiro.simple.data;

import lombok.AllArgsConstructor;
import lombok.SneakyThrows;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.IntFunction;
import java.util.stream.IntStream;

/**Подготовка датасета для MNIST*/
public class DatasetMNIST {

    @AllArgsConstructor
    public static class ExampleList{
        List<float[]> datasTrain;
        List<float[]> targetsTrain;
        List<float[]> datasTest;
        List<float[]> targetsTest;
    }

    /**Получить данные*/
    public static ExampleList getData(){
        List<float[]> datasTrain = new ArrayList<>();
        List<float[]> targetsTrain = new ArrayList<>();
        List<float[]> datasTest = new ArrayList<>();
        List<float[]> targetsTest = new ArrayList<>();

        prepareDataForMnist(targetsTrain, datasTrain, "d:\\Work\\Project\\0files\\mnist\\mnist_train.csv");
        prepareDataForMnist(targetsTest, datasTest, "d:\\Work\\Project\\0files\\mnist\\mnist_test.csv");

        return new ExampleList(datasTrain, targetsTrain, datasTest, targetsTest);
    }

    /**
     * Подготовка данных MNIST
     */
    @SneakyThrows
    private static void prepareDataForMnist(List<float[]> targets, List<float[]> datas, String fileName) {
        List<String> list = Files.readAllLines(Paths.get(fileName));
        float MAX = 1;//0.99;
        float MIN = 0;//0.01;
        for (String s : list) {
            String[] sarr = s.split(",");

            float[] target = new float[10];
            Arrays.fill(target, MIN);
            target[Integer.valueOf(sarr[0])] = MAX;
            targets.add(target);

            float[] data = new float[sarr.length - 1];
            for (int j = 1; j < sarr.length; j++)
                data[j - 1] = (Float.valueOf(sarr[j]) / 255.0F) * MAX + MIN;

            datas.add(data);
        }
    }

    /**Преобразуем списки входных и целевых векторов в массив примеров
     * Example[] examples = convertToExampleArray(
     *     inputs, targets,
     *     Example::new,   // ссылка на конструктор Example(float[], float[])
     *     Example[]::new
     * );*/
    private static <T> T[] convertToExampleArray(
            List<float[]> inputs,
            List<float[]> targets,
            BiFunction<float[], float[], T> creator,
            IntFunction<T[]> arrayCreator) {

        return IntStream.range(0, inputs.size())
                .mapToObj(i -> creator.apply(inputs.get(i), targets.get(i)))
                .toArray(arrayCreator);
    }
}
