package neiro.simple.net.layer;

import neiro.simple.net.Parameter;
import neiro.simple.net.Tensor;
import neiro.simple.net.layer.fun.LeakyReLULayer;
import neiro.simple.net.layer.fun.ReLULayer;

import java.util.List;

/**
 * Свёрточный слой (2D convolution) с использованием im2col + matmul.
 * <p>
 * Вход: 3D тензор [height, width, inChannels] (один пример) или
 *       4D тензор [batch, height, width, inChannels].<br>
 * Выход: соответственно 3D или 4D тензор с outChannels.
 * <p>
 * Поддерживает stride и padding (одинаковые по высоте и ширине).
 */
public class Conv2DLayer implements Layer {
    private final int inChannels;
    private final int outChannels;
    private final int kernelSize;
    private final int stride;
    private final int padding;

    // Параметры: ядро [kernelSize, kernelSize, inChannels, outChannels], смещение [outChannels]
    private Parameter kernel;
    private Parameter bias;

    // Кэш для backward
    private Tensor lastInput;
    private boolean inputWas3D; // флаг, что вход был 3D (без батча)

    public Conv2DLayer(int inChannels, int outChannels, int kernelSize, int stride, int padding) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.kernelSize = kernelSize;
        this.stride = stride;
        this.padding = padding;
    }

    @Override
    public void initializeParameters(Layer prev, Layer next) {
        kernel = new Parameter(kernelSize, kernelSize, inChannels, outChannels);
        bias = new Parameter(outChannels);

        // Выбор инициализации по типу следующего слоя
        if (next instanceof ReLULayer || next instanceof LeakyReLULayer) {
            kernel.initHe(true);
        } else {
            kernel.initXavier(true);
        }
        // bias остаётся нулевым
    }

    @Override
    public Tensor forward(Tensor input) {
        // Приводим к 4D, запоминая исходный ранг
        Tensor input4D;
        if (input.rank == 3) {
            // [H, W, C] -> [1, H, W, C]
            input4D = input.reshape(1, input.shape[0], input.shape[1], input.shape[2]);
            inputWas3D = true;
        } else if (input.rank == 4) {
            input4D = input;
            inputWas3D = false;
        } else {
            throw new IllegalArgumentException("Conv2DLayer: expected 3D or 4D input, got rank " + input.rank);
        }
        lastInput = input4D;

        int batch = input4D.shape[0];
        int inH = input4D.shape[1];
        int inW = input4D.shape[2];
        int kH = kernelSize, kW = kernelSize;

        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;

        // 1. im2col: [batch * outH * outW, kH * kW * inChannels]
        Tensor col = im2col(input4D);

        // 2. Ядро как матрица: [kH * kW * inChannels, outChannels]
        Tensor kernelMat = kernel.data.reshape(kH * kW * inChannels, outChannels);

        // 3. Матричное умножение
        Tensor outMat = Tensor.matmul(col, kernelMat); // [M, outChannels]

        // 4. Добавляем bias (broadcast по первому измерению)
        float[] b = bias.data.data;
        for (int i = 0; i < outMat.shape[0]; i++) {
            int rowOff = i * outChannels;
            for (int c = 0; c < outChannels; c++) {
                outMat.data[rowOff + c] += b[c];
            }
        }

        // 5. Возвращаем к исходной размерности
        Tensor out4D = outMat.reshape(batch, outH, outW, outChannels);
        if (inputWas3D) {
            // Убираем batch размерность
            return out4D.reshape(outH, outW, outChannels);
        } else {
            return out4D;
        }
    }

    /**
     * Преобразует входной 4D тензор в матрицу im2col.
     * Каждая строка соответствует одному положению фильтра и содержит все значения
     * рецептивного поля (kernelSize * kernelSize * inChannels).
     *
     * @param input входной тензор [batch, inH, inW, inC]
     * @return матрица [batch * outH * outW, kernelSize * kernelSize * inChannels]
     */
    private Tensor im2col(Tensor input) {
        int batch = input.shape[0];
        int inH = input.shape[1];
        int inW = input.shape[2];
        int inC = input.shape[3];
        int kH = kernelSize, kW = kernelSize;
        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;
        int patchSize = kH * kW * inC;

        Tensor col = new Tensor(batch * outH * outW, patchSize);
        int colRow = 0;

        for (int b = 0; b < batch; b++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int colOff = colRow * patchSize;
                    int p = 0;
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = oh * stride + kh - padding;
                        if (ih < 0 || ih >= inH) {
                            // padding нулями – пропускаем, значения остаются нулями
                            p += kW * inC;
                            continue;
                        }
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = ow * stride + kw - padding;
                            if (iw < 0 || iw >= inW) {
                                p += inC;
                                continue;
                            }
                            int inOff = ((b * inH + ih) * inW + iw) * inC;
                            for (int c = 0; c < inC; c++) {
                                col.data[colOff + p++] = input.data[inOff + c];
                            }
                        }
                    }
                    colRow++;
                }
            }
        }
        return col;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // gradOutput может быть 3D или 4D, приводим к 4D
        Tensor gradOut4D;
        if (inputWas3D) {
            // [outH, outW, outC] -> [1, outH, outW, outC]
            gradOut4D = gradOutput.reshape(1, gradOutput.shape[0], gradOutput.shape[1], gradOutput.shape[2]);
        } else {
            gradOut4D = gradOutput;
        }

        int batch = gradOut4D.shape[0];
        int outH = gradOut4D.shape[1];
        int outW = gradOut4D.shape[2];
        int outC = gradOut4D.shape[3];

        // 1. Градиент по bias: сумма по всем, кроме каналов
        float[] gradBias = bias.grad.data;
        for (int c = 0; c < outC; c++) gradBias[c] = 0.0f; // сбрасываем перед накоплением

        for (int b = 0; b < batch; b++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int off = ((b * outH + oh) * outW + ow) * outC;
                    for (int c = 0; c < outC; c++) {
                        gradBias[c] += gradOut4D.data[off + c];
                    }
                }
            }
        }

        // 2. Градиент по ядру через im2col и транспонирование
        Tensor gradOutMat = gradOut4D.reshape(batch * outH * outW, outC);
        Tensor col = im2col(lastInput); // [M, K] где K = kH*kW*inC
        Tensor colT = col.transpose(); // [K, M]
        Tensor gradKernelMat = Tensor.matmul(colT, gradOutMat); // [K, outC]

        // Прибавляем к накопленному градиенту ядра
        Tensor kernelGrad = kernel.grad.reshape(kernelSize * kernelSize * inChannels, outChannels);
        for (int i = 0; i < kernelGrad.size; i++) {
            kernelGrad.data[i] += gradKernelMat.data[i];
        }

        // 3. Градиент по входу: col2im
        Tensor kernelMat = kernel.data.reshape(kernelSize * kernelSize * inChannels, outChannels);
        Tensor kernelMatT = kernelMat.transpose(); // [outC, K]
        Tensor gradInputMat = Tensor.matmul(gradOutMat, kernelMatT); // [M, K]

        Tensor gradInput4D = col2im(gradInputMat, lastInput.shape);

        if (inputWas3D) {
            // Убираем batch размерность
            return gradInput4D.reshape(lastInput.shape[1], lastInput.shape[2], lastInput.shape[3]);
        } else {
            return gradInput4D;
        }
    }

    /**
     * Обратное im2col: распределяет градиенты из матрицы [M, patchSize] обратно в тензор
     * формы [batch, inH, inW, inC]. Значения суммируются в тех позициях, откуда были взяты.
     *
     * @param col        матрица градиентов [M, patchSize]
     * @param inputShape форма исходного входа [batch, inH, inW, inC]
     * @return тензор градиентов по входу
     */
    private Tensor col2im(Tensor col, int[] inputShape) {
        int batch = inputShape[0];
        int inH = inputShape[1];
        int inW = inputShape[2];
        int inC = inputShape[3];
        int kH = kernelSize, kW = kernelSize;
        int outH = (inH + 2 * padding - kH) / stride + 1;
        int outW = (inW + 2 * padding - kW) / stride + 1;
        int patchSize = kH * kW * inC;

        Tensor gradInput = new Tensor(inputShape);
        int colRow = 0;

        for (int b = 0; b < batch; b++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int colOff = colRow * patchSize;
                    int p = 0;
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = oh * stride + kh - padding;
                        if (ih < 0 || ih >= inH) {
                            p += kW * inC;
                            continue;
                        }
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = ow * stride + kw - padding;
                            if (iw < 0 || iw >= inW) {
                                p += inC;
                                continue;
                            }
                            int inOff = ((b * inH + ih) * inW + iw) * inC;
                            for (int c = 0; c < inC; c++) {
                                gradInput.data[inOff + c] += col.data[colOff + p++];
                            }
                        }
                    }
                    colRow++;
                }
            }
        }
        return gradInput;
    }

    @Override
    public List<Parameter> parameters() {
        return List.of(kernel, bias);
    }
}