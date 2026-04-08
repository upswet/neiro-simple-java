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

    /**
     * Косинусная близость между двумя векторами.
     * Возвращает значение от -1 до 1, где 1 — максимальное сходство.
     */
    public static double cosineSimilarity(double[] a, double[] b) {
        if (a.length != b.length) {
            throw new IllegalArgumentException("Векторы разной размерности");
        }
        double dot = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA == 0.0 || normB == 0.0) return 0.0;
        return dot / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Косинусное расстояние = 1 - косинусная близость.
     * Расстояние 0 означает идентичность, 2 — противоположность.
     */
    public static double cosineDistance(double[] a, double[] b) {
        return 1.0 - cosineSimilarity(a, b);
    }
}
