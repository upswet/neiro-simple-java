package neiro.simple.net;

import java.util.Arrays;

/**
 * Плоский (row-major) тензор с элементами типа float.
 * <p>
 * Данные хранятся в одномерном массиве {@code data}, размерности заданы в {@code shape},
 * шаги по измерениям вычислены в {@code strides}. Поддерживается до 4 измерений
 * (например, [batch, height, width, channels]).
 * <p>
 * Операции, изменяющие форму ({@code reshape}), не копируют данные, а создают
 * новое представление с теми же {@code data} и новыми {@code shape/strides}.
 * Операции, меняющие значения, создают новый тензор или модифицируют на месте.
 */
public class Tensor {

    /**
     * Одномерный массив, содержащий все элементы тензора в порядке row-major.
     * <p>
     * Пример: для тензора формы [2, 3] элементы лежат как [0,0], [0,1], [0,2], [1,0], [1,1], [1,2].
     */
    public final float[] data;

    /**
     * Размеры по каждому измерению. Например, [2, 3, 4] для 3D тензора.
     */
    public final int[] shape;

    /**
     * Шаги для перевода многомерных индексов в линейный индекс.
     * stride[i] = произведение размеров всех последующих измерений (shape[i+1] * shape[i+2] * ...).
     */
    public final int[] strides;

    /**
     * Количество измерений (rank), от 1 до 4.
     */
    public final int rank;

    /**
     * Общее количество элементов = произведение shape.
     */
    public final int size;

    // -------- Конструкторы -------------------------------------------------

    /**
     * Создаёт тензор заданной формы, заполненный нулями.
     *
     * @param shape размеры по измерениям
     */
    public Tensor(int... shape) {
        if (shape.length < 1 || shape.length > 4) {
            throw new IllegalArgumentException("Rank must be 1..4, got " + shape.length);
        }
        this.shape = shape.clone();
        this.rank = shape.length;
        this.size = prod(shape);
        this.data = new float[size];
        this.strides = computeStrides(shape);
    }

    /**
     * Создаёт тензор, используя готовый массив данных и заданную форму.
     * Массив НЕ копируется, используется переданная ссылка.
     *
     * @param data  одномерный массив значений (должен соответствовать форме)
     * @param shape размеры по измерениям
     * @throws IllegalArgumentException если длина data не равна произведению shape
     */
    public Tensor(float[] data, int... shape) {
        if (shape.length < 1 || shape.length > 4) {
            throw new IllegalArgumentException("Rank must be 1..4, got " + shape.length);
        }
        int prod = prod(shape);
        if (data.length != prod) {
            throw new IllegalArgumentException("Data length " + data.length + " != " + prod);
        }
        this.data = data;
        this.shape = shape.clone();
        this.rank = shape.length;
        this.size = prod;
        this.strides = computeStrides(shape);
    }

    /**
     * Создаёт глубокую копию тензора.
     *
     * @return новый тензор с теми же shape и скопированными данными
     */
    public Tensor copy() {
        return new Tensor(data.clone(), shape.clone());
    }

    // -------- Утилиты -------------------------------------------------------

    private static int prod(int[] arr) {
        int p = 1;
        for (int x : arr) p *= x;
        return p;
    }

    private static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    /**
     * Преобразует многомерные координаты в линейный индекс.
     *
     * @param indices массив длины rank с координатами
     * @return позиция в массиве data
     */
    public int index(int... indices) {
        if (indices.length != rank) {
            throw new IllegalArgumentException("Expected " + rank + " indices, got " + indices.length);
        }
        int idx = 0;
        for (int i = 0; i < rank; i++) {
            idx += indices[i] * strides[i];
        }
        return idx;
    }

    /**
     * Возвращает значение элемента по координатам.
     */
    public float get(int... indices) {
        return data[index(indices)];
    }

    /**
     * Устанавливает значение элемента по координатам.
     */
    public void set(float value, int... indices) {
        data[index(indices)] = value;
    }

    // -------- Изменение формы -----------------------------------------------

    /**
     * Возвращает новое представление тензора с изменённой формой.
     * Общее количество элементов должно совпадать. Данные не копируются.
     *
     * @param newShape новая форма
     * @return тензор с теми же data, но новой shape и strides
     */
    public Tensor reshape(int... newShape) {
        if (prod(newShape) != size) {
            throw new IllegalArgumentException("Total elements must remain " + size);
        }
        return new Tensor(data, newShape);
    }

    /**
     * Транспонирование для двумерного тензора (матрицы).
     *
     * @return новый тензор формы [shape[1], shape[0]]
     */
    public Tensor transpose() {
        if (rank != 2) {
            throw new UnsupportedOperationException("Transpose only supported for 2D tensors");
        }
        int rows = shape[0];
        int cols = shape[1];
        float[] transposed = new float[size];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j * rows + i] = data[i * cols + j];
            }
        }
        return new Tensor(transposed, cols, rows);
    }

    // -------- Поэлементные операции -----------------------------------------

    /**
     * Функциональный интерфейс для поэлементного преобразования.
     */
    @FunctionalInterface
    public interface ElementwiseOp {
        float apply(float x);
    }

    /**
     * Создаёт новый тензор, применяя операцию к каждому элементу.
     *
     * @param op операция
     * @return новый тензор той же формы
     */
    public Tensor map(ElementwiseOp op) {
        Tensor result = new Tensor(shape);
        for (int i = 0; i < size; i++) {
            result.data[i] = op.apply(data[i]);
        }
        return result;
    }

    /**
     * Применяет операцию к каждому элементу на месте (модифицирует текущий тензор).
     *
     * @param op операция
     */
    public void applyInPlace(ElementwiseOp op) {
        for (int i = 0; i < size; i++) {
            data[i] = op.apply(data[i]);
        }
    }

    // -------- Арифметика ----------------------------------------------------

    /**
     * Поэлементное сложение двух тензоров одинаковой формы.
     *
     * @return новый тензор = a + b
     */
    public static Tensor add(Tensor a, Tensor b) {
        if (!Arrays.equals(a.shape, b.shape)) {
            throw new IllegalArgumentException("Shapes must match: " +
                    Arrays.toString(a.shape) + " vs " + Arrays.toString(b.shape));
        }
        Tensor res = new Tensor(a.shape);
        for (int i = 0; i < res.size; i++) {
            res.data[i] = a.data[i] + b.data[i];
        }
        return res;
    }

    /**
     * Матричное умножение двух 2D тензоров: C = A * B.
     * A: [M, K], B: [K, N] -> C: [M, N].
     *
     * @return новый тензор
     */
    public static Tensor matmul(Tensor a, Tensor b) {
        if (a.rank != 2 || b.rank != 2) {
            throw new IllegalArgumentException("Both tensors must be 2D for matmul");
        }
        int m = a.shape[0];
        int k = a.shape[1];
        int k2 = b.shape[0];
        int n = b.shape[1];
        if (k != k2) {
            throw new IllegalArgumentException("Inner dimensions must agree: " + k + " vs " + k2);
        }
        Tensor c = new Tensor(m, n);
        for (int i = 0; i < m; i++) {
            int aRowOff = i * k;
            int cRowOff = i * n;
            for (int p = 0; p < k; p++) {
                float av = a.data[aRowOff + p];
                if (av == 0.0f) continue; // небольшая оптимизация
                int bRowOff = p * n;
                for (int j = 0; j < n; j++) {
                    c.data[cRowOff + j] += av * b.data[bRowOff + j];
                }
            }
        }
        return c;
    }
}
