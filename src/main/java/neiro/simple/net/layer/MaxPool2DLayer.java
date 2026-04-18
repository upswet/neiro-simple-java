package neiro.simple.net.layer;

import neiro.simple.net.Tensor;

/**
 * Слой максимального пулинга (Max Pooling 2D).
 * <p>
 * Вход: 3D [height, width, channels] или 4D [batch, height, width, channels].<br>
 * Выход: соответственно 3D или 4D тензор уменьшенных пространственных размеров.
 */
public class MaxPool2DLayer implements Layer {
    private final int poolSize;
    private final int stride;

    // Для backward нужно знать, где находились максимумы
    private int[] maxIndices; // линейный индекс внутри окна (0 .. poolSize*poolSize-1)
    private int[] lastInputShape;
    private boolean inputWas3D;

    public MaxPool2DLayer(int poolSize, int stride) {
        this.poolSize = poolSize;
        this.stride = stride;
    }

    @Override
    public void initializeParameters(Layer prev, Layer next) { /* нет параметров */ }

    @Override
    public Tensor forward(Tensor input) {
        // Приводим к 4D
        Tensor input4D;
        if (input.rank == 3) {
            input4D = input.reshape(1, input.shape[0], input.shape[1], input.shape[2]);
            inputWas3D = true;
        } else if (input.rank == 4) {
            input4D = input;
            inputWas3D = false;
        } else {
            throw new IllegalArgumentException("MaxPool2D: expected 3D or 4D input, got rank " + input.rank);
        }
        lastInputShape = input4D.shape.clone();

        int batch = input4D.shape[0];
        int inH = input4D.shape[1];
        int inW = input4D.shape[2];
        int channels = input4D.shape[3];

        int outH = (inH - poolSize) / stride + 1;
        int outW = (inW - poolSize) / stride + 1;

        Tensor output4D = new Tensor(batch, outH, outW, channels);
        maxIndices = new int[batch * outH * outW * channels];

        int outIdx = 0;
        for (int b = 0; b < batch; b++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    for (int c = 0; c < channels; c++) {
                        float maxVal = Float.NEGATIVE_INFINITY;
                        int maxPos = -1;
                        for (int ph = 0; ph < poolSize; ph++) {
                            int ih = oh * stride + ph;
                            for (int pw = 0; pw < poolSize; pw++) {
                                int iw = ow * stride + pw;
                                float val = input4D.get(b, ih, iw, c);
                                if (val > maxVal) {
                                    maxVal = val;
                                    maxPos = ph * poolSize + pw;
                                }
                            }
                        }
                        output4D.set(maxVal, b, oh, ow, c);
                        maxIndices[outIdx++] = maxPos;
                    }
                }
            }
        }

        if (inputWas3D) {
            // [1, outH, outW, C] -> [outH, outW, C]
            return output4D.reshape(outH, outW, channels);
        } else {
            return output4D;
        }
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // Приводим к 4D
        Tensor gradOut4D;
        int outH, outW, channels;
        if (inputWas3D) {
            outH = gradOutput.shape[0];
            outW = gradOutput.shape[1];
            channels = gradOutput.shape[2];
            gradOut4D = gradOutput.reshape(1, outH, outW, channels);
        } else {
            gradOut4D = gradOutput;
            outH = gradOut4D.shape[1];
            outW = gradOut4D.shape[2];
            channels = gradOut4D.shape[3];
        }

        int batch = lastInputShape[0];
        int inH = lastInputShape[1];
        int inW = lastInputShape[2];
        // channels уже совпадают

        Tensor gradInput4D = new Tensor(lastInputShape);
        int outIdx = 0;

        for (int b = 0; b < batch; b++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    for (int c = 0; c < channels; c++) {
                        int maxPos = maxIndices[outIdx++];
                        int ph = maxPos / poolSize;
                        int pw = maxPos % poolSize;
                        int ih = oh * stride + ph;
                        int iw = ow * stride + pw;
                        float grad = gradOut4D.get(b, oh, ow, c);
                        gradInput4D.data[gradInput4D.index(b, ih, iw, c)] += grad;
                    }
                }
            }
        }

        if (inputWas3D) {
            return gradInput4D.reshape(inH, inW, channels);
        } else {
            return gradInput4D;
        }
    }
}