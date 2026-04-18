package neiro.simple.net;


import neiro.simple.net.dataloader.DataLoader;
import neiro.simple.net.loss.Loss;
import neiro.simple.net.metric.Metric;
import neiro.simple.net.optimaizer.Optimizer;

import java.util.*;

/**
 * Управляет процессом обучения модели.
 */
public class Trainer {
    private Model model;
    private volatile boolean stopRequested = false;
    private final List<Metric> metrics = new ArrayList<>();

    public void addMetric(Metric metric) {
        metrics.add(metric);
    }

    public void fit(Model model, Optimizer optimizer, Loss loss,
                    DataLoader trainData, Callback callback,
                    DataLoader validationData, int batchSize, int epochs) {
        this.model = model;
        stopRequested = false;
        model.setTraining(true);

        callback.onTrainBegin(this);

        for (int epoch = 0; epoch < epochs && !stopRequested; epoch++) {
            callback.onEpochBegin(this, epoch);
            long epochStart = System.nanoTime();

            for (Metric m : metrics) m.reset();

            trainData.reset();
            Iterator<DataLoader.Example> it = trainData.iterator();
            float epochLoss = 0.0f;
            int batchCount = 0;

            while (it.hasNext() && !stopRequested) {
                // Собираем батч
                List<Tensor> batchInputs = new ArrayList<>();
                List<Object> batchTargets = new ArrayList<>();

                for (int i = 0; i < batchSize && it.hasNext(); i++) {
                    DataLoader.Example ex = it.next();
                    batchInputs.add(ex.inputs());
                    batchTargets.add(ex.targets());
                }
                if (batchInputs.isEmpty()) break;

                // Stack inputs в один тензор
                Tensor batchInput = stackTensors(batchInputs);
                // Для target пока оставляем как есть (one-hot или индексы)
                Object batchTarget = batchTargets.size() == 1 ? batchTargets.get(0) : stackTargets(batchTargets);

                callback.onBatchBegin(this, epoch, batchCount);

                // Forward
                Tensor output = model.forward(batchInput);

                // Loss
                float batchLoss = loss.compute(output, (Tensor) batchTarget);
                epochLoss += batchLoss;
                batchCount++;

                // Backward
                Tensor gradOutput = loss.gradient(output, (Tensor) batchTarget);
                model.backward(gradOutput);

                // Обновление весов
                optimizer.update(model.parameters());

                // Метрики
                for (Metric m : metrics) m.update(output, batchTarget);

                callback.onBatchEnd(this, epoch, batchCount - 1, batchLoss);
            }

            float avgTrainLoss = batchCount > 0 ? epochLoss / batchCount : 0.0f;
            Map<String, Float> logs = new HashMap<>();
            logs.put("loss", avgTrainLoss);

            if (validationData != null) {
                float valLoss = evaluate(validationData, loss, batchSize);
                logs.put("val_loss", valLoss);
            }

            for (Metric m : metrics) {
                logs.put(m.name(), m.getValue());
            }
            logs.put("duration", (System.nanoTime() - epochStart) / 1_000_000f);
            logs.put("epoch", (float) epoch);

            callback.onEpochEnd(this, epoch, logs);
        }

        model.setTraining(false);
        callback.onTrainEnd(this);
    }

    /**Вычислить функцию потерь на тестовой выборке*/
    private float evaluate(DataLoader data, Loss loss, int batchSize) {
        data.reset();
        Iterator<DataLoader.Example> it = data.iterator();
        float totalLoss = 0.0f;
        int batchCount = 0;

        while (it.hasNext()) {
            List<Tensor> batchInputs = new ArrayList<>();
            List<Object> batchTargets = new ArrayList<>();

            for (int i = 0; i < batchSize && it.hasNext(); i++) {
                DataLoader.Example ex = it.next();
                batchInputs.add(ex.inputs());
                batchTargets.add(ex.targets());
            }
            if (batchInputs.isEmpty()) break;

            Tensor batchInput = stackTensors(batchInputs);
            Object batchTarget = batchTargets.size() == 1 ? batchTargets.get(0) : stackTargets(batchTargets);

            Tensor output = model.forward(batchInput);
            totalLoss += loss.compute(output, (Tensor) batchTarget);
            batchCount++;
        }
        return batchCount > 0 ? totalLoss / batchCount : 0.0f;
    }

    private Tensor stackTensors(List<Tensor> tensors) {
        // Все тензоры должны иметь одинаковую форму
        Tensor first = tensors.get(0);
        int[] shape = first.shape;
        int rank = shape.length;
        int[] newShape = new int[rank + 1];
        newShape[0] = tensors.size();
        System.arraycopy(shape, 0, newShape, 1, rank);

        int perSample = first.size;
        float[] data = new float[tensors.size() * perSample];
        int off = 0;
        for (Tensor t : tensors) {
            System.arraycopy(t.data, 0, data, off, perSample);
            off += perSample;
        }
        return new Tensor(data, newShape);
    }

    private Tensor stackTargets(List<Object> targets) {
        // Предполагаем, что все target - это Tensor (one-hot)
        List<Tensor> tensors = new ArrayList<>();
        for (Object obj : targets) {
            if (!(obj instanceof Tensor)) throw new IllegalArgumentException("Targets must be Tensors");
            tensors.add((Tensor) obj);
        }
        return stackTensors(tensors);
    }

    public void stopTraining() {
        stopRequested = true;
    }
}
