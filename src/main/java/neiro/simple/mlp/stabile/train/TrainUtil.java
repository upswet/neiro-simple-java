package neiro.simple.mlp.stabile.train;

/**Утилиты для тренировки нейросети*/
public class TrainUtil {
    /**Интерфейс для тестирования нейросети (функция оценки)*/
    public interface Estimation {
        /**
         * Проверка результата вычислений во время тестирования
         *
         * @param expectedVec     - вектор ожидаемых значений
         * @param outputVec       - вектор выходных значений
         * @param acceptableError - допустимая ошибка при которой ответ считается верным
         * @return - правильный ответ или нет
         */
        boolean process(double[] outputVec, double[] expectedVec, double acceptableError);
    }

    /**вернёт индекс максимального элемента из вектора*/
    private static Integer findMax(double[] arr) {
        double max = -999F;
        Integer imax = -1;
        for (int i = 0; i < arr.length; i++)
            if (arr[i] > max) {
                max = arr[i];
                imax = i;
            }
        return imax;
    }

    /**Функция эстимации (можно ли считать данный ответ нейросети правильным). По максимальному значению*/
    public static Estimation estimationMax = (double[] outputVec, double[] expectedVec, double acceptableError) ->{
        return findMax(outputVec).equals(findMax(expectedVec));
    };

    /**Функция эстимации (можно ли считать данный ответ нейросети правильным). Функция потерь*/
    public static Estimation estimationLoss = (double[] outputVec, double[] expectedVec, double acceptableError) ->{
        double loss=0;
        for(int i=0; i<outputVec.length;i++)
            loss = loss + Math.abs(expectedVec[i]-outputVec[i]);
        return loss < acceptableError;
    };

}
