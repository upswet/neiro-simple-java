package neiro.simple.mlp.stabile.model;

import java.io.Serializable;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.function.UnaryOperator;

/**Класс содержит функции активации, их производные, функции инициализации весов и производные функции потерь*/
public class Fun {
    /**Функции активации и их производные*/

    //сигмоида
    public static UnaryOperator<Double> SIGMOID = (UnaryOperator<Double> & Serializable) x -> 1.0 / (1 + Math.exp(-x));
    public static BiFunction<Double, Double, Double> SIGMOID_DERIVATIVE =  (BiFunction<Double, Double, Double> & Serializable) (iValue, oValue) -> (
            //SIGMOID.apply(iValue) * (1 - SIGMOID.apply(iValue)) /// классический вариант
            oValue * (1 - oValue) // ускоренный вариант
    );

    //тангес
    public static UnaryOperator<Double> TANH = (UnaryOperator<Double> & Serializable) x -> Math.tanh(x);
    public static BiFunction<Double, Double, Double> TANH_DERIVATIVE = (BiFunction<Double, Double, Double> & Serializable) (iValue, oValue) -> 1.0 - oValue * oValue; // производная tanh = 1 - tanh²(x)

    //релу
    public static UnaryOperator<Double> RELU = (UnaryOperator<Double> & Serializable) x -> x > 0 ? x : 0.01 * x;
    public static BiFunction<Double, Double, Double> RELU_DERIVATIVE = (BiFunction<Double, Double, Double> & Serializable) (iValue, oValue) -> iValue > 0 ? 1.0 : 0.01;

    //HardTanh
    public static UnaryOperator<Double> HARDTANH = (UnaryOperator<Double> & Serializable) x -> Math.max(-1, Math.min(1, x));
    public static BiFunction<Double, Double, Double> HARDTANH_DERIVATIVE = (BiFunction<Double, Double, Double> & Serializable) (iValue, oValue) ->  (iValue >= -1.0 && iValue <= 1.0) ? 1.0 : 0.0;


    /**Функции инициализации весов
     * std - максимальный вес
     * fanIn - число входов в слой
     * fanOut - число выходов из слоя*/

    //нормальное распределение
    public static Supplier<Double> INIT_NORMAL(double std){
        return (Supplier<Double> & Serializable) () -> {
            // Генерация случайного числа из нормального распределения
            double u1 = Math.random();
            double u2 = Math.random();
            double z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            return z * std;
        };
    }

    //инициализация хавьера
    public static Supplier<Double> INIT_XAVIER (int fanIn, int fanOut) {
        return (Supplier<Double> & Serializable) () -> {
            double std = Math.sqrt(2.0 / (fanIn + fanOut));
            return INIT_NORMAL(std).get();
        };
    }

    //инициализация he
    public static Supplier<Double> INIT_HE(int fanIn) {
        return (Supplier<Double> & Serializable) () -> {
            double std = Math.sqrt(2.0 / fanIn);
            return INIT_NORMAL(std).get();
        };
    }

    /**Производные функции потерь*/

    //Mean Squared Error (MSE) — среднеквадратичная ошибка
    public static BiFunction<Double, Double, Double> LOSS_DERIVATIVE_MSE = (BiFunction<Double, Double, Double> & Serializable) (oValue, target) -> oValue - target; //для  MSE  loss = (target - output)^2 производная по output: 2*(output - target) (но обычно берут (output - target)

    //Mean Absolute Error (MAE) — средняя абсолютная ошибка
    public static BiFunction<Double, Double, Double> LOSS_DERIVATIVE_MAE = (BiFunction<Double, Double, Double> & Serializable) (oValue, target) ->  oValue == target ? 1 : oValue < target ? -0.5 : 0.5;
}
