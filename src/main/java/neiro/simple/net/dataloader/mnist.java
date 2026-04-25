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
import neiro.simple.net.loss.NegativeSamplingWithEmbeddingLoss;
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
        //cnn(args);
        negative_sampling();
    }

    public static void negative_sampling(){
        // 1. Строим модель, которая возвращает скрытое представление
        Model model = new Model()
                .addLayer(new DenseLayer(784, 256))
                .addLayer(new ReLULayer())
                .addLayer(new DenseLayer(256, 128))   // hiddenDim = 128
                .initializeParameters();

        // 2. Создаём Negative Sampling Loss с эмбеддингами
        int numClasses = 10;
        int hiddenDim = 128;
        int numNegativeSamples = 5;
        NegativeSamplingWithEmbeddingLoss loss = new NegativeSamplingWithEmbeddingLoss(
                numClasses, hiddenDim, numNegativeSamples
        );

        // 3. Добавляем обучаемые параметры loss в модель
        model.addExternalParameters(loss.parameters());

        // 4. Оптимизатор
        Optimizer optimizer = new Adam(0.001f);

        // 5. Trainer работает без изменений
        Trainer trainer = new Trainer();
        //trainer.addMetric(new CategoricalAccuracyMetric());  - эта метрика не адаптирована для работы со скрытым слоем

        DataLoader trainLoader = new DataLoaderMnist("d:\\Work\\Project\\0files\\mnist\\mnist_train.csv", false);
        DataLoader testLoader  = new DataLoaderMnist("d:\\Work\\Project\\0files\\mnist\\mnist_test.csv", false);

        trainer.fit(model, optimizer, loss, trainLoader, logger, testLoader, 32, 3);

        // 6. Ручное вычисление точности на тестовом наборе
        System.out.println("Evaluating final accuracy on test set...");
        int correct = 0;
        int total = 0;
        testLoader.reset();
        for (DataLoader.Example ex : testLoader) {
            Tensor input = ex.inputs();
            // Приводим к батчу размером 1, если надо
            //if (input.rank == 3) {
            //    input = input.reshape(1, input.shape[0], input.shape[1], input.shape[2]);
            //}
            Tensor hidden = model.forward(input);       // [1, hiddenDim]

            // Определяем истинную метку
            int trueLabel;
            Object targetObj = ex.targets();
            if (targetObj instanceof Tensor) {
                Tensor t = (Tensor) targetObj;
                trueLabel = NegativeSamplingWithEmbeddingLoss.argmax(t.data, 0, t.size);
            } else {
                trueLabel = ((int[]) targetObj)[0];
            }

            // Считаем логиты для всех 10 классов
            float[] scores = new float[numClasses];
            for (int c = 0; c < numClasses; c++) {
                scores[c] = loss.computeScoreForClass(hidden, 0, c);
            }
            int predLabel = NegativeSamplingWithEmbeddingLoss.argmax(scores, 0, scores.length);

            if (predLabel == trueLabel) correct++;
            total++;
        }
        float accuracy = (float) correct / total;
        System.out.printf("Final test accuracy: %.4f (%d/%d)%n", accuracy, correct, total);
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
