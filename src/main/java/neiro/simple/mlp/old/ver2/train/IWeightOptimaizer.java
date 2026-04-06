package neiro.simple.mlp.old.ver2.train;

import lombok.AllArgsConstructor;
import lombok.Setter;
import neiro.simple.mlp.old.ver2.WeightWrapper;

import java.util.function.Function;

/**Оптимизатор для весовых коэффициентов*/
public interface IWeightOptimaizer {

    /**Обработка
     * @param weight - весовой коэффициент. Может быть изменён при необходимости
     * @param grad - градиент изменения*/
    void update(WeightWrapper weight, double grad);

    /**Пересчёт состояния оптимизатора после успешного обновления весовых коэффициентов в данной итерации*/
    default void nextStep(){};

    /**Сбросить внутреннее состояние оптимизатора */
    default void reset(){};

    /**Получить подходящий под конкретный оптимизатор WeightWrapper*/
    Function<Double, WeightWrapper> getWeightWrapper();

    /**Простейший (классический) оптимизатор весовых коэффициентов*/
    @AllArgsConstructor
    public static class classic implements IWeightOptimaizer{
        /**Гиперпараметры обучения*/
        double learningRate=0.02; //коэффициент скорости обучения

        @Override public void update(WeightWrapper weight, double grad) {weight.item = weight.item - learningRate * grad;}

        @Override public Function<Double, WeightWrapper> getWeightWrapper() {return WeightWrapper.classic::new;}
    }

    /**Оптимизатор весовых коэффициентов с простейшим моментом*/
    @AllArgsConstructor
    public static class momentum implements IWeightOptimaizer{
        /**Гиперпараметры обучения*/
        double learningRate; //коэффициент скорости обучения
        double momentumCoef =0.9; //коэффициент момента (коэф инерции)

        @Override public void update(WeightWrapper weight, double grad) {
            WeightWrapper.momentum wrapper = (WeightWrapper.momentum) weight;
            wrapper.m = wrapper.m * momentumCoef + learningRate * grad;
            wrapper.item = wrapper.item - wrapper.m;
        }

        @Override public Function<Double, WeightWrapper> getWeightWrapper() {return WeightWrapper.momentum::new;}
    }

    /**Оптимизатор весовых коэффициентов по методу Нестерова (усовершеннствованный мтеод моменента) NAG*/
    @AllArgsConstructor
    public static class nag implements IWeightOptimaizer{
        /**Гиперпараметры обучения*/
        double learningRate; //коэффициент скорости обучения
        double momentumCoef =0.9; //коэффициент момента (коэф инерции)

        @Override public void update(WeightWrapper weight, double grad) {
            WeightWrapper.momentum wrapper = (WeightWrapper.momentum) weight;

            double prevM = wrapper.m; // Сохраняем предыдущее значение момента (до обновления)

            wrapper.m = wrapper.m * momentumCoef + learningRate * grad;
            wrapper.item = wrapper.item - ((1 + momentumCoef) * wrapper.m - momentumCoef * prevM); // Обновляем вес по формуле NAG: w = w - ((1 + μ) * m_new - μ * m_old)
        }

        @Override public Function<Double, WeightWrapper> getWeightWrapper() {return WeightWrapper.momentum::new;}
    }

    /**Оптимизатор весовых коэффициентов - ADAM*/
    public static class adam implements IWeightOptimaizer{
        /**Гиперпараметры оптимизатора ADAM*/
        double learningRate=0.001; //коэффициент скорости обучения
        @Setter
        public double beta1 = 0.9;
        @Setter public double beta2 = 0.999;
        @Setter public double epsilon = 1e-8;
        /**Параметры для ускорения расчёта*/
        double beta1Pow;
        double beta2Pow;
        double invCorrection1;
        double invCorrection2;

        public adam(double learningRate) {
            this.learningRate = learningRate;
            init();
        }

        public adam(double learningRate, double beta1, double beta2, double epsilon) {
            this.learningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
            init();
        }

        /**Инициализация начальных значений*/
        private void init() {
            beta1Pow = 1.0;
            beta2Pow = 1.0;
            invCorrection1 = 1.0 / (1.0 - beta1); // для первого шага
            invCorrection2 = 1.0 / (1.0 - beta2);
        }

        @Override
        public void nextStep(){
            beta1Pow *= beta1;
            beta2Pow *= beta2;
            invCorrection1 = 1.0 / (1.0 - beta1Pow);
            invCorrection2 = 1.0 / (1.0 - beta2Pow);
        }
        @Override
        public  void reset(){
            beta1Pow = beta1;
            beta2Pow = beta2;
            invCorrection1 = 1.0 / (1.0 - beta1Pow);
            invCorrection2 = 1.0 / (1.0 - beta2Pow);
        };

        @Override public void update(WeightWrapper weight, double grad) {
            WeightWrapper.adam wrapper = (WeightWrapper.adam) weight;
            // Обновление моментов Adam
            wrapper.m = beta1 * wrapper.m + (1 - beta1) * grad;
            wrapper.v = beta2 * wrapper.v + (1 - beta2) * grad * grad;
            // Смещение моментов (bias correction)
            double mHat = wrapper.m * invCorrection1;
            double vHat = wrapper.v * invCorrection2 ;
            // Обновление веса
            wrapper.item = wrapper.item - learningRate * mHat / (Math.sqrt(vHat) + epsilon);
        }

        @Override public Function<Double, WeightWrapper> getWeightWrapper() {return WeightWrapper.adam::new;}
    }
}
