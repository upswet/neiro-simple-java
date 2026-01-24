package neiro.simple.obj;

/**Входной слой*/
public class LayerInput extends Layer {
    /**Конструктор входного слоя
     * @param nCount - количество нейронов в входном слою
     * @return - входной слой*/
    public LayerInput(int nCount){
        for(int i =0; i<nCount; i++)
            this.neirons.add(Neiron.createInputNeiron());
    }

    /**Задать вектор входных данных*/
    private void setInputs(double[] inputs){
        assert (inputs.length!=neirons.size()) : "Несовпадение размерности";

        for(int i=0; i<inputs.length; i++)
            neirons.get(i).oValue=inputs[i];
    }

    /**Прямой проход для нейронов слоя (вычисление)
     * @param inputs - вектор входных данных*/
    public void forward(double[] inputs){
        setInputs(inputs);
    }

    /**Обратное распространение ошибки (корректировка весов). Может запускаться только после выполнения прямого распространения*/
    public void backward(){
        // Ничего не делаем так как исходящие из входного слоя связи уже обновлены первым промежуточным слоем, а дельты вычислять не надо так как входящих связей нет
    }
}
