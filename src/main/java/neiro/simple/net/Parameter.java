package neiro.simple.net;

import java.util.Arrays;
import java.util.Random;

/**
 * Обучаемый параметр нейронной сети.
 * <p>
 * Содержит значения параметра ({@code data}) и накопленный градиент ({@code grad}).
 * Поддерживает стандартные стратегии инициализации весов (He, Xavier).
 */
public class Parameter {

    /** Тензор со значениями параметра (веса или смещения). */
    public final Tensor data;

    /** Тензор с накопленными градиентами, имеет ту же форму, что и data. */
    public final Tensor grad;

    /**
     * Создаёт параметр заданной формы, заполненный нулями.
     */
    public Parameter(int... shape) {
        this.data = new Tensor(shape);
        this.grad = new Tensor(shape);
    }

    /**
     * Создаёт параметр, используя готовый тензор данных.
     * Градиенты инициализируются нулями той же формы.
     */
    public Parameter(Tensor data) {
        this.data = data;
        this.grad = new Tensor(data.shape);
    }

    /**
     * Обнуляет все накопленные градиенты.
     */
    public void zeroGrad() {
        Arrays.fill(grad.data, 0.0f);
    }

    /**
     * Инициализация He (Kaiming) для ReLU-подобных активаций.
     * fanIn вычисляется как произведение всех измерений, кроме последнего.
     *
     * @param uniform если true – равномерное распределение, иначе нормальное
     */
    public void initHe(boolean uniform) {
        int fanIn = 1;
        for (int i = 0; i < data.rank - 1; i++) {
            fanIn *= data.shape[i];
        }
        Random rnd = new Random();
        if (uniform) {
            float limit = (float) Math.sqrt(6.0 / fanIn);
            for (int i = 0; i < data.size; i++) {
                data.data[i] = (rnd.nextFloat() * 2 - 1) * limit;
            }
        } else {
            float std = (float) Math.sqrt(2.0 / fanIn);
            for (int i = 0; i < data.size; i++) {
                data.data[i] = (float) rnd.nextGaussian() * std;
            }
        }
    }

    /**
     * Инициализация Xavier (Glorot).
     * fanIn – произведение всех измерений, кроме последнего,
     * fanOut – размер последнего измерения.
     *
     * @param uniform если true – равномерное, иначе нормальное
     */
    public void initXavier(boolean uniform) {
        int fanIn = 1;
        for (int i = 0; i < data.rank - 1; i++) {
            fanIn *= data.shape[i];
        }
        int fanOut = data.shape[data.rank - 1];
        Random rnd = new Random();
        if (uniform) {
            float limit = (float) Math.sqrt(6.0 / (fanIn + fanOut));
            for (int i = 0; i < data.size; i++) {
                data.data[i] = (rnd.nextFloat() * 2 - 1) * limit;
            }
        } else {
            float std = (float) Math.sqrt(2.0 / (fanIn + fanOut));
            for (int i = 0; i < data.size; i++) {
                data.data[i] = (float) rnd.nextGaussian() * std;
            }
        }
    }
}
