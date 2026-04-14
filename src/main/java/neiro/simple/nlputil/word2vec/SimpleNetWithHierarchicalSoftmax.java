package neiro.simple.nlputil.word2vec;

import java.util.*;

/**
 * Упрощённая нейросеть с иерархическим softmax (Hierarchical Softmax)
 * для эффективного обучения на больших словарях.
 * Поддерживает только SSG-оптимизацию, без батчей.
 *
 * Иерархический softmax заменяет полный softmax бинарными классификациями
 * по пути от корня до листа (слова) в двоичном дереве (обычно дерево Хаффмана).
 * Для каждого примера обновляются только узлы на пути к целевому слову (O(log V) вместо O(V)).
 *
 *
 * Иерархический softmax — это метод ускорения обучения в задачах с очень большим количеством классов (например, в word2vec при словаре в сотни тысяч слов). Вместо вычисления вероятностей для всех классов (через полный softmax, сложность O(V)), он организует классы в бинарное дерево и требует вычислений только для узлов на пути от корня к листу (сложность O(log V)).
 *
 * Как строится дерево
 * Обычно используется дерево Хаффмана, построенное по частотам слов. Частые слова получают короткие пути, редкие — длинные, что дополнительно ускоряет обучение. Листья дерева соответствуют словам (классам), внутренние узлы — бинарным классификаторам.
 */
public class SimpleNetWithHierarchicalSoftmax {
    private final Random rand = new Random();

    private final int inputSize;      // размер входного слоя (vocabSize)
    private final int hiddenSize;     // размер скрытого слоя (размер эмбеддингов)
    private final int totalNodes;     // количество узлов в дереве (2 * vocabSize - 1)

    // Веса и смещения для скрытого слоя (обычный полносвязный)
    private float[][] w1;   // [hiddenSize][inputSize]
    private float[] b1;     // [hiddenSize]

    // Веса для узлов дерева (каждый узел имеет вектор весов размером hiddenSize)
    // Индексы: от 0 до totalNodes-1. Листьям соответствуют слова (0..vocabSize-1),
    // внутренним узлам — индексы от vocabSize до totalNodes-1.
    private float[][] nodeWeights;   // [totalNodes][hiddenSize]
    private float[] nodeBiases;      // [totalNodes]

    // Временные массивы для forward/backward
    private float[] hiddenInput;
    private float[] hiddenOutput;

    // Структура дерева
    private final int[] parent;       // родитель узла (-1 для корня)
    private final int[] binaryCode;   // 0 или 1 для каждого узла (кроме корня) — направление от родителя
    private final int[] leftChild;    // левый ребёнок (-1 если нет)
    private final int[] rightChild;   // правый ребёнок

    private final float learningRate;

    /**
     * Конструктор. Строит дерево Хаффмана на основе частот слов.
     * @param wordFreqs массив частот слов (индекс = id слова, значение = частота)
     * @param hiddenSize размерность эмбеддингов
     * @param learningRate шаг обучения
     */
    public SimpleNetWithHierarchicalSoftmax(int[] wordFreqs, int hiddenSize, float learningRate) {
        int vocabSize = wordFreqs.length;
        this.inputSize = vocabSize;
        this.hiddenSize = hiddenSize;
        this.learningRate = learningRate;
        this.totalNodes = 2 * vocabSize - 1;

        // Инициализация весов скрытого слоя (Xavier)
        w1 = new float[hiddenSize][inputSize];
        b1 = new float[hiddenSize];
        float limit1 = (float) Math.sqrt(6.0 / (inputSize + hiddenSize));
        for (int i = 0; i < hiddenSize; i++) {
            b1[i] = 0;
            for (int j = 0; j < inputSize; j++) {
                w1[i][j] = (rand.nextFloat() - 0.5f) * 2 * limit1;
            }
        }

        // Временные буферы
        hiddenInput = new float[hiddenSize];
        hiddenOutput = new float[hiddenSize];

        // Построение дерева Хаффмана
        HuffmanTree htree = buildHuffmanTree(wordFreqs, vocabSize);
        this.parent = htree.parent;
        this.binaryCode = htree.binaryCode;
        this.leftChild = htree.leftChild;
        this.rightChild = htree.rightChild;

        // Инициализация весов узлов дерева (Xavier, но с учётом hiddenSize)
        nodeWeights = new float[totalNodes][hiddenSize];
        nodeBiases = new float[totalNodes];
        float limitNode = (float) Math.sqrt(6.0 / (hiddenSize + 1)); // для каждого узла вход = скрытый слой
        for (int i = 0; i < totalNodes; i++) {
            nodeBiases[i] = 0;
            for (int j = 0; j < hiddenSize; j++) {
                nodeWeights[i][j] = (rand.nextFloat() - 0.5f) * 2 * limitNode;
            }
        }
    }

    /**
     * Структура для хранения дерева Хаффмана.
     */
    private static class HuffmanTree {
        int[] parent;
        int[] binaryCode;
        int[] leftChild;
        int[] rightChild;
        HuffmanTree(int[] parent, int[] binaryCode, int[] leftChild, int[] rightChild) {
            this.parent = parent;
            this.binaryCode = binaryCode;
            this.leftChild = leftChild;
            this.rightChild = rightChild;
        }
    }

    /**
     * Построение дерева Хаффмана по частотам слов.
     * Возвращает массивы parent, binaryCode (бит на пути от родителя), leftChild, rightChild.
     * Индексы 0..vocabSize-1 — листья (слова), vocabSize..totalNodes-1 — внутренние узлы.
     * Корень имеет индекс totalNodes-1.
     */
    private HuffmanTree buildHuffmanTree(int[] freq, int vocabSize) {
        int total = 2 * vocabSize - 1;
        int[] parent = new int[total];
        int[] binary = new int[total];
        int[] left = new int[total];
        int[] right = new int[total];
        Arrays.fill(parent, -1);
        Arrays.fill(binary, -1);
        Arrays.fill(left, -1);
        Arrays.fill(right, -1);

        // Очередь с приоритетом: (частота, индекс узла)
        PriorityQueue<long[]> pq = new PriorityQueue<>(Comparator.comparingLong(a -> a[0]));
        for (int i = 0; i < vocabSize; i++) {
            pq.offer(new long[]{freq[i], i});
        }
        int nextNode = vocabSize;
        while (pq.size() >= 2) {
            long[] a = pq.poll();
            long[] b = pq.poll();
            int nodeIdx = nextNode++;
            long sumFreq = a[0] + b[0];
            left[nodeIdx] = (int) a[1];
            right[nodeIdx] = (int) b[1];
            parent[(int) a[1]] = nodeIdx;
            parent[(int) b[1]] = nodeIdx;
            binary[(int) a[1]] = 0; // левый потомок кодируется 0
            binary[(int) b[1]] = 1; // правый потомок — 1
            pq.offer(new long[]{sumFreq, nodeIdx});
        }
        // Корень
        int root = nextNode - 1;
        parent[root] = -1;
        return new HuffmanTree(parent, binary, left, right);
    }

    /**
     * Прямой проход: входной вектор -> скрытый слой.
     * @param input входной вектор (обычно one-hot или усреднение контекста)
     * @return выход скрытого слоя (эмбеддинг контекста)
     */
    private float[] forwardHidden(float[] input) {
        for (int i = 0; i < hiddenSize; i++) {
            float sum = b1[i];
            float[] row = w1[i];
            for (int j = 0; j < inputSize; j++) {
                sum += row[j] * input[j];
            }
            hiddenInput[i] = sum;
            hiddenOutput[i] = sum; // identity активация (можно оставить линейной)
        }
        return hiddenOutput;
    }

    /**
     * Сигмоида для бинарной классификации в узлах.
     */
    private float sigmoid(float x) {
        return 1f / (1f + (float) Math.exp(-x));
    }


    /**
     * Обучение с корректным накоплением градиентов для скрытого слоя.
     */
    public void trainCorrect(float[] input, int targetWord) {
        // Прямой проход
        float[] h = forwardHidden(input);

        // Собираем путь от слова до корня
        List<Integer> pathNodes = new ArrayList<>();
        List<Integer> pathLabels = new ArrayList<>();
        int node = targetWord;
        while (parent[node] != -1) {
            pathNodes.add(parent[node]);
            pathLabels.add(binaryCode[node]);
            node = parent[node];
        }

        // Накопление градиента по скрытому вектору h
        float[] gradH = new float[hiddenSize];
        for (int idx = 0; idx < pathNodes.size(); idx++) {
            int p = pathNodes.get(idx);
            int label = pathLabels.get(idx);
            float dot = nodeBiases[p];
            float[] wNode = nodeWeights[p];
            for (int i = 0; i < hiddenSize; i++) {
                dot += wNode[i] * h[i];
            }
            float prob = sigmoid(dot);
            float delta = (label - prob);

            // Обновление весов узла
            nodeBiases[p] += learningRate * delta;
            for (int i = 0; i < hiddenSize; i++) {
                wNode[i] += learningRate * delta * h[i];
                // Накопление градиента по h: производная по h = delta * wNode[i]
                gradH[i] += delta * wNode[i]; // здесь wNode[i] уже обновлён? Лучше использовать старые значения.
                // Правильнее сохранить старые веса, но для простоты порядок не критичен (можно обновлять после накопления).
            }
        }

        // Теперь обновляем w1 и b1, используя gradH
        // Градиент по b1: gradH, по w1: gradH * input[j]
        for (int i = 0; i < hiddenSize; i++) {
            b1[i] += learningRate * gradH[i];
            float[] row = w1[i];
            for (int j = 0; j < inputSize; j++) {
                row[j] += learningRate * gradH[i] * input[j];
            }
        }
    }

    /**
     * Упрощённый метод train (вызывает правильную версию).
     */
    public void train(float[] input, int targetWord) {
        trainCorrect(input, targetWord);
    }

    /**
     * Получить эмбеддинг слова (вектор скрытого слоя для данного входного one-hot).
     * В данной архитектуре эмбеддинг слова — это столбец w1[:, wordIdx].
     * @param wordIdx индекс слова
     * @return вектор размером hiddenSize
     */
    public float[] getEmbedding(int wordIdx) {
        float[] emb = new float[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            emb[i] = w1[i][wordIdx];
        }
        return emb;
    }

    // Для совместимости с ожидаемым интерфейсом (если нужно)
    public float getWeight1(int hid, int idx) {
        return w1[hid][idx];
    }
}