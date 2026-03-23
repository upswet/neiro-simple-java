package neiro.simple.mlp.stabile.dto;

import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;

/**Результат обучения и тестирования нейросети*/
@FieldDefaults(level = AccessLevel.PUBLIC)
public class TrainResult{
    Long duration = 0L; //время обучения
    int epoch=0;        //количество эпох обучения
    double quality= 0F; //качество решения на тестовом наборе

    /**Установить значение*/
    public void set(Long duration, int epoch, double quality) {
        this.duration = duration;
        this.epoch = epoch;
        this.quality = quality;
    }

    @Override
    public String toString() {
        return "TrainResult{" +
                "duration=" + duration +
                ", epoch=" + epoch +
                ", quality=" + quality +
                '}';
    }
}
