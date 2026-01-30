package neiro.simple.obj;

/**Входной слой*/
public class LayerInput extends Layer {
    double dropoutRate = 0.0; // процент дропаута (0.0 - нет дропаута, 0.5 - 50%)

    /**Конструктор входного слоя
     * @param nCount - количество нейронов в входном слою
     * @param dropoutRate - процент дропаута (от 0.0 до 0.99)
     * @return - входной слой*/
    public LayerInput(int nCount, double dropoutRate){
        this.dropoutRate = dropoutRate;

        for(int i =0; i<nCount; i++)
            this.neirons.add(Neiron.createInputNeiron());
    }

    /**Прямой проход для нейронов слоя (вычисление)
     * @param inputs - вектор входных данных
     * @param trainingMode - если истина, то режим обучения, иначе режим работы*/
    public void forward(double[] inputs, boolean trainingMode){
        assert (inputs.length!=neirons.size()) : "Несовпадение размерности";

        for(int i=0; i<inputs.length; i++) {
            neirons.get(i).oValue = inputs[i];
            if (trainingMode && dropoutRate>0)
                if (Math.random() < dropoutRate)
                    neirons.get(i).oValue = Math.random();

        }
    }

    /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения*/
    public void backward(){
        // Ничего не делаем так как исходящие из входного слоя связи уже обновлены первым промежуточным слоем, а дельты вычислять не надо так как входящих связей нет
    }
}
