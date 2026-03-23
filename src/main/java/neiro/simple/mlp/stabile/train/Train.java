package neiro.simple.mlp.stabile.train;

import lombok.extern.slf4j.Slf4j;
import neiro.simple.mlp.stabile.dto.Example;
import neiro.simple.mlp.stabile.dto.Examples;
import neiro.simple.mlp.stabile.dto.TrainResult;
import neiro.simple.mlp.stabile.model.Net;

import java.time.Duration;
import java.time.Instant;

/**Класс для обучения нейросети*/
@Slf4j
public class Train {
    /**Протестировать эффективность нейросети на примере тестовых данных
     * @param net - нейросеть типа mlp
     * @param data - данные для тестирования
     * @param estimation - функция оценки качества ответа нейросети
     * @param acceptableError - допустимая ошибка при которой ответ нейросети всё равно считается правильным
     * @return - доля правильных ответов*/
    public static double test(Net net, Example[] data, TrainUtil.Estimation estimation, double acceptableError){
        int success=0;
        for(int i=0; i< data.length; i++){
            double[] actual = net.forward(data[i].input());
            if (estimation.process(actual,data[i].target(), acceptableError))
                success++;
        }

        return (double) success / data.length;
    }

    /**Обучить нейросеть
     * @param net - нейросеть типа mlp
     * @param epoch - количество эпох обучения
     * @param data - данные для обучения и тестирования
     * @param estimation - функция оценки качества ответа нейросети
     * @param acceptableError - допустимая ошибка при которой ответ нейросети всё равно считается правильным
     * @param optimaizer - оптимизатор весовых коэффициентов
     * @param batchSize - размер пачки. Минус один если не используем пакетное обучение
     * @param periodStepPrint - периодичность (в шагах) когда выводить промежуточные результаты
     * @return - доля результаты обучения и тестирования*/
    public static TrainResult train(Net net, int epoch, Examples data, TrainUtil.Estimation estimation, double acceptableError, IWeightOptimaizer optimaizer, int batchSize, int periodStepPrint){
        net.init(optimaizer.getWeightWrapper()); //инициализация весовых коэффициентов согласно используемому оптимизатору

        TrainResult result = new TrainResult();
        for(int i=1; i<=epoch; i++){
            result.set(
                    trainEpoch(net, data.trainData(), optimaizer, batchSize, periodStepPrint),
                    i,
                    test(net, data.testData(), estimation, acceptableError)
            );
            log.info(result.toString());
            if (result.quality >= 1 - acceptableError) {
                log.info("Required accuracy has been obtained: {}", result.quality);
                break;
            }
        }

        return result;
    }

    /**Тренировка в пределах одной эпохи
     * @param net - нейросеть типа mlp
     * @param data - данные для обучения
     * @param optimaizer - оптимизатор весовых коэффициентов
     * @param batchSize - размер пачки. Минус один если не используем пакетное обучение
     * @param periodStepPrint - периодичность (в шагах) когда выводить промежуточные результаты
     * @return - длитлеьность обучения в пределах одной эпохи*/
    private static long trainEpoch(Net net, Example[] data, IWeightOptimaizer optimaizer, int batchSize, int periodStepPrint){
        int printCount = periodStepPrint; //сколько шагов осталось до печати промежуточных значений
        int batchSizeCount = 1; //для пакетного режима обработки. Номер текущего примера в пакете

        Instant start = Instant.now();
        for(int i=0; i<data.length; i++, printCount--){
            if (printCount==0){
                log.info("step {} / {}. Duration {}", i, data.length, Duration.between(start, Instant.now()).toMillis());
                printCount = periodStepPrint;
            }

            if (batchSize == -1)
                trainExampleNoBatch(net, data[i], optimaizer); //без пакетного режима
            else
                batchSizeCount = trainExampleBatch(net, data[i], optimaizer, batchSizeCount, batchSize, i == (data.length -1)); //в пакетном режиме
        }
        return Duration.between(start, Instant.now()).toMillis();
    }


    /**Тренировка в пределах одного примера с пакетной обработкой
     * @param net - нейросеть типа mlp
     * @param data - данные для обучения
     * @param optimaizer - оптимизатор весовых коэффициентов
     * @param isEndEpoch - признак конца эпохи
     * @param batchSizeCount - Только для пакетного режима обучения. Текущий размер пачки*/
    private static int trainExampleBatch(Net net, Example data, IWeightOptimaizer optimaizer, int batchSizeCount, int batchSize, boolean isEndEpoch){
        double[] output = net.forward(data.input());
        boolean isEndBatch = batchSizeCount == batchSize; //достигли конца пакета
        net.backward(data.target(),optimaizer,true, isEndEpoch || isEndBatch, batchSizeCount);
        return isEndBatch ? 0 : batchSizeCount+1;
    }


    /**Тренировка в пределах одного примера без пакетной обработки
     * @param net - нейросеть типа mlp
     * @param data - данные для обучения
     * @param optimaizer - оптимизатор весовых коэффициентов*/
    private static void trainExampleNoBatch(Net net, Example data, IWeightOptimaizer optimaizer){
        double[] output = net.forward(data.input());
        net.backward(data.target(),optimaizer,false, false, -1);
    }

}
