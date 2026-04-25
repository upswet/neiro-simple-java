package neiro.simple.net.layer;

import lombok.Setter;
import neiro.simple.net.Tensor;
import java.util.Random;

/**
 * Слой Dropout для регуляризации нейронной сети.
 * <p>
 * Во время обучения (training = true) случайно обнуляет долю входных нейронов,
 * заданную вероятностью {@code p}. Оставшиеся нейроны масштабируются на
 * {@code 1/(1-p)}, чтобы сохранить среднее ожидаемое значение.
 * <p>
 * В режиме предсказания (training = false) слой не изменяет входные данные.
 * <p>
 * Пример использования: после полносвязного или свёрточного слоя перед активацией.
 * <pre>{@code
 * Model model = new Model()
 *     .addLayer(new DenseLayer(784, 256))
 *     .addLayer(new ReLULayer())
 *     .addLayer(new DropoutLayer(0.5f))   // выключаем 50% нейронов
 *     .addLayer(new DenseLayer(256, 10))
 *     .initializeParameters();
 * }</pre>
 */
public class DropoutLayer implements Layer {
    private final float p;               // вероятность обнуления нейрона
    private final float scale;           // множитель для выживших нейронов
    /**
     * -- SETTER --
     *  Устанавливает режим работы слоя.
     *
     * @param training true – обучение (dropout активен), false – предсказание (прямой проход)
     */
    @Setter
    private boolean training;            // текущий режим (обучение / предсказание)
    private float[] mask;                // маска, использованная в последнем forward
    private final Random random;

    /**
     * @param p вероятность обнуления одного нейрона (обычно 0.1–0.5)
     */
    public DropoutLayer(float p) {
        if (p < 0f || p >= 1f) {
            throw new IllegalArgumentException("p must be in [0, 1)");
        }
        this.p = p;
        this.scale = 1.0f / (1.0f - p);
        this.training = true;   // по умолчанию слой в режиме обучения
        this.random = new Random();
    }

    public DropoutLayer() {
        this(0.5f);
    }

    @Override
    public Tensor forward(Tensor input) {
        if (!training) {
            return input;   // в режиме предсказания ничего не меняем
        }

        mask = new float[input.size];
        for (int i = 0; i < mask.length; i++) {
            mask[i] = random.nextFloat() < p ? 0.0f : scale;
        }

        float[] outData = new float[input.size];
        for (int i = 0; i < outData.length; i++) {
            outData[i] = input.data[i] * mask[i];
        }
        return new Tensor(outData, input.shape.clone());
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        if (!training) {
            return gradOutput;   // в режиме предсказания градиент не изменяется
        }
        // Градиент проходит только через выжившие нейроны (с учётом масштаба)
        float[] gradInputData = new float[gradOutput.size];
        for (int i = 0; i < gradInputData.length; i++) {
            gradInputData[i] = gradOutput.data[i] * mask[i];
        }
        return new Tensor(gradInputData, gradOutput.shape.clone());
    }

    @Override
    public void initializeParameters(Layer prev, Layer next) {
        // нет параметров
    }

    /**
     * Возвращает текущую маску (может использоваться для отладки).
     */
    public float[] getLastMask() {
        return mask;
    }
}