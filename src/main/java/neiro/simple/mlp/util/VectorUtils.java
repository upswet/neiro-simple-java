package neiro.simple.mlp.util;

public class VectorUtils {
    // скалярное произведение двух векторов
    public static double dotProduct(double[] a, double[] b) {
        assert a.length!=b.length : "Несовпадение размерности!";

        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    // расстояние между двумя векторами
    public static double cosineSimilarity(double[] a, double[] b) {
        assert a.length!=b.length : "Несовпадение размерности!";

        double dot = dotProduct(a, b);
        double normA = 0, normB = 0;
        for (int i = 0; i < a.length; i++) {
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }
}
