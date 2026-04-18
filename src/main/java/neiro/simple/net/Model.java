package neiro.simple.net;

import lombok.Getter;
import lombok.Setter;
import lombok.experimental.Accessors;
import neiro.simple.net.layer.Layer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Accessors(chain = true)
public class Model {
    private final List<Layer> layers = new ArrayList<>();
    @Getter @Setter
    private boolean training = true;

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

    public List<Parameter> parameters() {
        List<Parameter> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.parameters());
        }
        return Collections.unmodifiableList(params);
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
