package neiro.simple.net.dataloader;

import lombok.SneakyThrows;
import neiro.simple.net.Tensor;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

/**
 * Загрузчик данных MNIST из CSV файла.
 * Поддерживает два режима:
 * <ul>
 *   <li>для MLP – каждый входной пример возвращается как тензор формы [784]</li>
 *   <li>для CNN – каждый входной пример возвращается как тензор формы [28, 28, 1]</li>
 * </ul>
 * Целевые метки возвращаются в виде one‑hot тензора [10].
 */
public class DataLoaderMnist implements DataLoader {
    private final List<Tensor> inputs = new ArrayList<>();
    private final List<Tensor> targets = new ArrayList<>();
    private final int size;
    private final boolean forCnn;

    /**
     * Создаёт загрузчик данных из CSV файла.
     *
     * @param pathToFileCsv путь к файлу (например, "mnist_train.csv")
     * @param forCnn если true, входной тензор будет формы [28, 28, 1],
     *               иначе [784] (одномерный)
     */
    @SneakyThrows
    public DataLoaderMnist(String pathToFileCsv, boolean forCnn) {
        this.forCnn = forCnn;
        List<String> lines = Files.readAllLines(Paths.get(pathToFileCsv));

        for (String line : lines) {
            String[] parts = line.split(",");
            int label = Integer.parseInt(parts[0]);

            // Целевой one‑hot тензор [10]
            float[] targetData = new float[10];
            targetData[label] = 1.0f;
            targets.add(new Tensor(targetData, 10));

            // Входные пиксели (нормализованные в [0,1])
            float[] pixelData = new float[parts.length - 1];
            for (int i = 1; i < parts.length; i++) {
                pixelData[i - 1] = Float.parseFloat(parts[i]) / 255.0f;
            }

            Tensor inputTensor;
            if (forCnn) {
                // Для CNN: [28, 28, 1]
                inputTensor = new Tensor(28, 28, 1);
                for (int i = 0; i < 784; i++) {
                    int y = i / 28;
                    int x = i % 28;
                    inputTensor.set(pixelData[i], y, x, 0);
                }
            } else {
                // Для MLP: [784]
                inputTensor = new Tensor(pixelData, 784);
            }
            inputs.add(inputTensor);
        }

        this.size = inputs.size();
    }

    @Override
    public void reset() {
        // Порядок не важен, итератор всегда начинает с начала
    }

    @Override
    public Iterator<Example> iterator() {
        return new Iterator<Example>() {
            private int idx = 0;

            @Override
            public boolean hasNext() {
                return idx < size;
            }

            @Override
            public Example next() {
                if (!hasNext()) throw new NoSuchElementException();
                Tensor input = inputs.get(idx);
                Tensor target = targets.get(idx);
                idx++;
                return new Example() {
                    @Override
                    public Tensor inputs() {
                        return input;
                    }

                    @Override
                    public Object targets() {
                        return target; // one-hot Tensor
                    }
                };
            }
        };
    }

    /**
     * Количество примеров в наборе данных.
     */
    public int size() {
        return size;
    }
}
