package neiro.simple.net.dataloader;

import neiro.simple.net.Tensor;

public interface DataLoader extends Iterable<DataLoader.Example> {
    void reset();
    interface Example {
        Tensor inputs();
        Object targets(); // Tensor или int[] и т.п.
    }
}
