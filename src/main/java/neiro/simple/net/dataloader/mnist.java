package neiro.simple.net.dataloader;


import lombok.extern.slf4j.Slf4j;
import neiro.simple.net.Callback;
import neiro.simple.net.Model;
import neiro.simple.net.Tensor;
import neiro.simple.net.Trainer;
import neiro.simple.net.layer.Conv2DLayer;
import neiro.simple.net.layer.DenseLayer;
import neiro.simple.net.layer.FlattenLayer;
import neiro.simple.net.layer.MaxPool2DLayer;
import neiro.simple.net.layer.fun.ReLULayer;
import neiro.simple.net.layer.fun.SoftmaxLayer;
import neiro.simple.net.loss.CategoricalCrossEntropyLoss;
import neiro.simple.net.loss.Loss;
import neiro.simple.net.metric.CategoricalAccuracyMetric;
import neiro.simple.net.optimaizer.Adam;
import neiro.simple.net.optimaizer.Optimizer;

import java.util.Map;

@Slf4j
public class mnist {

    static Callback logger = new Callback() {
        @Override
        public void onEpochEnd(Trainer trainer, int epoch, Map<String, Float> logs) {
            for (Map.Entry entry : logs.entrySet())
                log.info(entry.getKey().toString() + " : " + entry.getValue().toString());

            log.info("=======================");
        }
    };

    public static void main(String[] args) {
        //mlp(args);
        cnn(args);
    }


    public static void cnn(String[] args) {
        // 1. Загрузка данных (forCnn = true)
        DataLoader trainLoader = new DataLoaderMnist("d:\\Work\\Project\\0files\\mnist\\mnist_train.csv", true);
        DataLoader testLoader  = new DataLoaderMnist("d:\\Work\\Project\\0files\\mnist\\mnist_test.csv", true);

        // 2. Построение модели CNN
        Model model = new Model()
                // Первый свёрточный блок
                .addLayer(new Conv2DLayer(1, 8, 3, 1, 1))   // 28x28x1 -> 28x28x8  // [b,28,28,1] -> [b,28,28,8]
                .addLayer(new ReLULayer())
                .addLayer(new MaxPool2DLayer(2, 2))         // 28x28x8 -> 14x14x8  // -> [b,14,14,8]
                // Второй свёрточный блок
                .addLayer(new Conv2DLayer(8, 16, 3, 1, 1))  // 14x14x8 -> 14x14x16  // -> [b,14,14,16]
                .addLayer(new ReLULayer())
                .addLayer(new MaxPool2DLayer(2, 2))         // 14x14x16 -> 7x7x16    // -> [b,7,7,16]
                // Полносвязная часть
                .addLayer(new FlattenLayer())               // 7*7*16 = 784     // -> [b,784]
                .addLayer(new DenseLayer(784, 128))
                .addLayer(new ReLULayer())
                .addLayer(new DenseLayer(128, 10))
                .addLayer(new SoftmaxLayer())
                .initializeParameters();

        // 3. Оптимизатор и функция потерь
        Optimizer optimizer = new Adam(0.001f, 0.9f, 0.999f, 1e-8f);
        Loss loss = new CategoricalCrossEntropyLoss();

        // 4. Тренер и метрики
        Trainer trainer = new Trainer();
        trainer.addMetric(new CategoricalAccuracyMetric());

        // 6. Запуск обучения
        trainer.fit(model, optimizer, loss, trainLoader, logger, testLoader, 32, 2);

        // 7. Финальная оценка
        clacFinalMetric(testLoader, model);
    }


    public static void mlp(String[] args) {
        // 1. Загрузка данных
        DataLoader trainLoader = new DataLoaderMnist("d:\\Work\\Project\\0files\\mnist\\mnist_train.csv", false);
        DataLoader testLoader  = new DataLoaderMnist("d:\\Work\\Project\\0files\\mnist\\mnist_test.csv", false);

        // 2. Построение модели MLP: 784 -> 128 -> 10
        Model model = new Model()
                .addLayer(new DenseLayer(784, 128))
                .addLayer(new ReLULayer())
                .addLayer(new DenseLayer(128, 10))
                .addLayer(new SoftmaxLayer())
                .initializeParameters();

        // 3. Оптимизатор и функция потерь
        Optimizer optimizer = new Adam(0.001f, 0.9f, 0.999f, 1e-8f);
        Loss loss = new CategoricalCrossEntropyLoss();

        // 4. Тренер и метрики
        Trainer trainer = new Trainer();
        trainer.addMetric(new CategoricalAccuracyMetric());

        // 6. Запуск обучения
        trainer.fit(model, optimizer, loss, trainLoader, logger, testLoader, 32, 2);

        // 7. Финальная оценка на тесте
        clacFinalMetric(testLoader, model);
    }

    private static void clacFinalMetric(DataLoader testLoader, Model model) {
        CategoricalAccuracyMetric finalMetric = new CategoricalAccuracyMetric();
        testLoader.reset();
        for (DataLoader.Example ex : testLoader) {
            Tensor pred = model.forward(ex.inputs());
            finalMetric.update(pred, ex.targets());
        }
        System.out.printf("Final test accuracy: %.4f%n", finalMetric.getValue());
    }
}
