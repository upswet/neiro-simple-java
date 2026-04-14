package neiro.simple.nlputil.word2vec; // или в отдельный пакет

import java.util.Random;

/**
 * Максимально упрощённая полносвязная нейросеть с одним скрытым слоем,
 * поддерживающая только SSG-оптимизацию (стохастический градиентный спуск),
 * обучение без батчей (online) и negative sampling на выходном слое.
 *
 *
 * Negative Sampling — это метод ускорения обучения нейросетей для задач с большим количеством классов (например, word2vec).
 *
 * Что это: Вместо обновления весов для всех выходных нейронов (тысячи/миллионы классов) при каждом примере, мы обновляем только веса для:
 * одного положительного класса (правильный ответ)
 * небольшого количества отрицательных классов (случайно выбранные неверные ответы)
 * Остальным классам веса не меняются.
 *
 * Зачем:
 * Резко снижает вычислительную сложность — вместо O(V) операций на пример становится O(1) (V — размер словаря).
 * Позволяет обучаться на больших словарях (сотни тысяч слов) без использования полного softmax.
 * Даёт качество, близкое к полному softmax при правильно выбранном количестве отрицательных сэмплов (обычно 5–20).
 *
 * альтернатива - иерархический софтмакс (с деревом хаффмана на выходном слое)
 */
public class SimpleNetWithNegativeSampling {

    // Вспомогательный класс для хранения примера для SimpleNet
    public static class Example {
        float[] input;
        int positiveClass;
        int[] negativeClasses;
        Example(float[] input, int positiveClass, int[] negativeClasses) {
            this.input = input;
            this.positiveClass = positiveClass;
            this.negativeClasses = negativeClasses;
        }
    }

    private final Random rand = new Random();
    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;
    private float[][] w1;  // [hiddenSize][inputSize]
    private float[] b1;    // [hiddenSize]
    private float[][] w2;  // [outputSize][hiddenSize]
    private float[] b2;    // [outputSize]
    private float[] hiddenInput;
    private float[] hiddenOutput;
    private float[] finalInput;
    private float[] finalOutput;
    private final float learningRate;

    public SimpleNetWithNegativeSampling(int inputSize, int hiddenSize, int outputSize, int negCount, float learningRate) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;
        w1 = new float[hiddenSize][inputSize];
        b1 = new float[hiddenSize];
        w2 = new float[outputSize][hiddenSize];
        b2 = new float[outputSize];
        float limit1 = (float) Math.sqrt(6.0 / (inputSize + hiddenSize));
        for (int i = 0; i < hiddenSize; i++) {
            b1[i] = 0;
            for (int j = 0; j < inputSize; j++) {
                w1[i][j] = (rand.nextFloat() - 0.5f) * 2 * limit1;
            }
        }
        float limit2 = (float) Math.sqrt(6.0 / (hiddenSize + outputSize));
        for (int i = 0; i < outputSize; i++) {
            b2[i] = 0;
            for (int j = 0; j < hiddenSize; j++) {
                w2[i][j] = (rand.nextFloat() - 0.5f) * 2 * limit2;
            }
        }
        hiddenInput = new float[hiddenSize];
        hiddenOutput = new float[hiddenSize];
        finalInput = new float[outputSize];
        finalOutput = new float[outputSize];
    }

    private float relu(float x) { return x > 0 ? x : 0; }
    private float reluDeriv(float x) { return x > 0 ? 1 : 0; }
    private float sigmoid(float x) { return 1f / (1f + (float) Math.exp(-x)); }

    public float[] forward(float[] input) {
        for (int i = 0; i < hiddenSize; i++) {
            float sum = b1[i];
            for (int j = 0; j < inputSize; j++) sum += w1[i][j] * input[j];
            hiddenInput[i] = sum;
            hiddenOutput[i] = relu(sum);
        }
        for (int i = 0; i < outputSize; i++) {
            float sum = b2[i];
            for (int j = 0; j < hiddenSize; j++) sum += w2[i][j] * hiddenOutput[j];
            finalInput[i] = sum;
            finalOutput[i] = sigmoid(sum);
        }
        return finalOutput;
    }

    // Обучение с явно заданными отрицательными классами
    public void train(float[] input, int positiveClass, int[] negativeClasses) {
        forward(input);

        // Дельта выходного слоя только для положительного и отрицательных классов
        float[] deltaOut = new float[outputSize];
        deltaOut[positiveClass] = finalOutput[positiveClass] - 1.0f;
        for (int neg : negativeClasses) {
            deltaOut[neg] = finalOutput[neg] - 0.0f;
        }

        // Дельта скрытого слоя
        float[] deltaHidden = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            float sum = 0;
            for (int k = 0; k < outputSize; k++) {
                if (deltaOut[k] != 0) sum += deltaOut[k] * w2[k][i];
            }
            deltaHidden[i] = sum * reluDeriv(hiddenInput[i]);
        }

        // Обновление w2 и b2 (только для затронутых выходных нейронов)
        for (int k = 0; k < outputSize; k++) {
            if (deltaOut[k] == 0) continue;
            b2[k] -= learningRate * deltaOut[k];
            for (int j = 0; j < hiddenSize; j++) {
                w2[k][j] -= learningRate * deltaOut[k] * hiddenOutput[j];
            }
        }

        // Обновление w1 и b1 (полностью)
        for (int i = 0; i < hiddenSize; i++) {
            b1[i] -= learningRate * deltaHidden[i];
            for (int j = 0; j < inputSize; j++) {
                w1[i][j] -= learningRate * deltaHidden[i] * input[j];
            }
        }
    }

    // Для извлечения эмбеддингов (вес от входного нейрона idx к скрытому нейрону hid)
    public float getWeight1(int hid, int idx) {
        return w1[hid][idx];
    }
}