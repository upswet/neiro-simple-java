package neiro.simple.mlp;

/**Утилиты для работы с нейросетью*/
public class NetUtils {

    /**вернёт индекс максимального элемента из вектора*/
    public static Integer findMax(float[] arr) {
        double max = -999F;
        Integer imax = -1;
        for (int i = 0; i < arr.length; i++)
            if (arr[i] > max) {
                max = arr[i];
                imax = i;
            }
        return imax;
    }

    /**Функции инициализации весов через нормальное распределение
     * std - максимальный веся*/
    public static float initNormal(double std){
            // Генерация случайного числа из нормального распределения
            double u1 = Math.random();
            double u2 = Math.random();
            double z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            return (float) (z * std);
    }

    /**
     * Преобразует вектор вещественных чисел в one-hot вектор.
     * One-hot вектор имеет длину, равную длине входного вектора,
     * единицу на позиции максимального элемента и нули в остальных.
     *
     * @param vector входной вектор (не null, не пустой)
     * @return one-hot вектор (float[])
     * @throws IllegalArgumentException если vector == null или vector.length == 0
     */
    public static float[] toOneHot(float[] vector) {
        if (vector == null || vector.length == 0)
            throw new IllegalArgumentException("Вектор не должен быть null или пустым");

        // Находим индекс максимального значения
        int maxIndex = 0;
        for (int i = 1; i < vector.length; i++)
            if (vector[i] > vector[maxIndex])
                maxIndex = i;

        // Создаём one-hot вектор
        float[] oneHot = new float[vector.length];
        oneHot[maxIndex] = 1.0f;
        return oneHot;
    }

    /**
     * Вычисляет softmax
     * Преобразует вектор чисел (например, выходов нейросети) в вероятностное распределение:
     *      Все значения становятся в интервале (0,1).
     *      Сумма всех выходов равна 1.
     *      Большие входные числа дают большие вероятности (но нелинейно, за счёт экспоненты).
     * Главная цель — выбрать наиболее вероятный класс в задачах многоклассовой классификации
     *
     * @param input входной вектор (не null, не пустой)
     * @return вектор вероятностей (float[]), сумма элементов = 1.0
     * @throws IllegalArgumentException если input == null или input.length == 0
     */
    public static float[] toSoftmax(float[] input) {
        if (input == null || input.length == 0)
            throw new IllegalArgumentException("Входной вектор не должен быть null или пустым");

        // Находим максимальное значение для стабильности
        float max = input[0];
        for (int i = 1; i < input.length; i++)
            if (input[i] > max)
                max = input[i];

        // Вычисляем экспоненты (exp(x_i - max)) и сумму
        double[] expValues = new double[input.length];
        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            expValues[i] = Math.exp(input[i] - max);
            sum += expValues[i];
        }

        // Нормируем и возвращаем float[]
        float[] result = new float[input.length];
        for (int i = 0; i < input.length; i++) {
            result[i] = (float) (expValues[i] / sum);
        }
        return result;
    }
}
