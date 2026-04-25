package neiro.simple.net;

import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;
import neiro.simple.net.layer.Layer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

@Accessors(chain = true)
public class Model {
    private final List<Layer> layers = new ArrayList<>();

    @Getter
    private boolean training = true;

    /**Установить или снять режим тренировки*/
    public void setTraining(boolean training) {
        this.training = training;
        for (Layer layer : layers)
            layer.setTraining(training);
    }


    private final List<Parameter> externalParameters = new ArrayList<>();

    public Tensor forward(Tensor input) {
        Tensor current = input;
        for (Layer layer : layers) {
            current = layer.forward(current);
        }
        return current;
    }

    public void backward(Tensor gradOutput) {
        Tensor grad = gradOutput;
        for (int i = layers.size() - 1; i >= 0; i--) {
            grad = layers.get(i).backward(grad);
        }
    }

    /**Передаёт параметры модели (всех её слоёв) оптимизатору чтобы они обучались, то есть чтобы к ним применялись вычисленные градиенты
     * Для параметров слоёв градиенты вычисляются при обратном проходе. Для внешних параметров градиенты должны вычисляться кастомно*/
    public List<Parameter> parameters() {
        List<Parameter> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.parameters());
        }
        params.addAll(externalParameters);
        return Collections.unmodifiableList(params);
    }

    /**
     * Добавляет внешний обучаемый параметр (например, из функции потерь).
     * Эти параметры будут включены в список, возвращаемый {@link #parameters()}.
     *
     * @param param параметр для добавления
     * @return this (для цепочечных вызовов)
     */
    public Model addExternalParameter(Parameter param) {
        externalParameters.add(param);
        return this;
    }

    /**
     * Добавляет коллекцию внешних обучаемых параметров.
     *
     * @param params коллекция параметров
     * @return this
     */
    public Model addExternalParameters(Collection<Parameter> params) {
        externalParameters.addAll(params);
        return this;
    }

    public List<Layer> layers() {
        return Collections.unmodifiableList(layers);
    }

    public Model addLayer(Layer layer) {
        layers.add(layer);
        return this;
    }

    public Model initializeParameters() {
        for (int i = 0; i < layers.size(); i++) {
            Layer prev = i > 0 ? layers.get(i - 1) : null;
            Layer next = i < layers.size() - 1 ? layers.get(i + 1) : null;
            layers.get(i).initializeParameters(prev, next);
        }
        return this;
    }
}
