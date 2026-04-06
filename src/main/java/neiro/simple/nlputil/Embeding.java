package neiro.simple.nlputil;

import neiro.simple.mlp.old.ver2.model.Fun;

/*
Эмбеддинг — это плотный вектор фиксированной размерности, который ставится в соответствие каждому токену. В процессе обучения модели эти векторы настраиваются так, чтобы семантически близкие токены были близки в векторном пространстве.

эмбеддинги — это просто матрица весов [vocab_size, emb_dim], где каждая строка — вектор токена. Их можно обучить независимо от полной языковой модели, используя задачу типа Word2Vec (Skip-gram + negative sampling). Для этого не нужна сложная нейросеть — достаточно прямого SGD-обновления.
Ниже — полностью самодостаточная реализация на чистой Java без библиотек. Она создаст массив эмбеддингов для ваших токенов и обучит его на текстовом корпусе.
*/
public class Embeding {

    /**Получить массив эмбедингов размерности  [vocabSize][embedDim] заполненные случайными значениями
     * @param vocabSize - размер словаря токенов
     * @param embedDim - размерность эмбединга
     * @return массив эмбедингов размерности vocabSize который потребуется дообучить согласно словарю*/
    private static double[][] create(int vocabSize, int embedDim){
        double[][] embeddings = new double[vocabSize][embedDim];
        for (int i = 0; i < vocabSize; i++)
            for (int j = 0; j < embedDim; j++)
                embeddings[i][j] = Fun.INIT_XAVIER(vocabSize, embedDim).get();
        return  embeddings;
    }
}
