package neiro.simple.nlputil.word2vec;

import lombok.extern.slf4j.Slf4j;

import java.util.*;
import java.util.stream.Collectors;

/**Утилиты. Подсчёт расстояния между векторами и близости слов по векторам и ещё операции с векторами*/
@Slf4j
public class Word2VecUtils {
    /**
     * Косинусная близость между двумя векторами.
     * Возвращает значение от -1 до 1, где 1 — максимальное сходство.
     */
    public static double cosineSimilarity(float[] a, float[] b) {
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
    public static double cosineDistance(float[] a, float[] b) {
        return 1.0 - cosineSimilarity(a, b);
    }

    /**
     * Поиск topK ближайших слов к заданному слову по косинусной близости.
     * Возвращает список пар (слово, близость), отсортированный по убыванию близости.
     * Само слово исключается из результата.
     */
    public static List<Map.Entry<String, Double>> findNearestWords(
            Map<String, float[]> wordVectors,
            String word,
            int topK) {
        float[] targetVec = wordVectors.get(word);
        if (targetVec == null) {
            //throw new IllegalArgumentException("Слово '" + word + "' не найдено в словаре");
            log.info("Word {} not found in dict", word);
            return List.of();
        }

        List<Map.Entry<String, Double>> similarities = new ArrayList<>();
        for (Map.Entry<String, float[]> entry : wordVectors.entrySet()) {
            if (entry.getKey().equals(word)) continue;
            double sim = cosineSimilarity(targetVec, entry.getValue());
            similarities.add(new AbstractMap.SimpleEntry<>(entry.getKey(), sim));
        }

        similarities.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        List result = similarities.stream().limit(topK).collect(Collectors.toList());
        log.info("word={}, list={}", word,  result.toString());
        return result;
    }

    /**
     * Пример использования: для заданных слов выводит косинусное расстояние.
     */
    public static void printCosineDistance(Map<String, float[]> vectors, String word1, String word2) {
        float[] v1 = vectors.get(word1);
        float[] v2 = vectors.get(word2);
        if (v1 == null || v2 == null) {
            System.out.println("Одно из слов не найдено");
            return;
        }
        double sim = cosineSimilarity(v1, v2);
        double dist = 1.0 - sim;
        System.out.printf("Cosine proximity between '%s' и '%s': %.4f\n", word1, word2, sim);
        System.out.printf("Cosine distance: %.4f\n", dist);
    }

    /**
     * Сложение двух векторов (покомпонентное).
     */
    public static float[] add(float[] a, float[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("Vectors of different length");
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) result[i] = a[i] + b[i];
        return result;
    }

    /**
     * Вычитание векторов: a - b.
     */
    public static float[] subtract(float[] a, float[] b) {
        if (a.length != b.length) throw new IllegalArgumentException("Vectors of different length");
        float[] result = new float[a.length];
        for (int i = 0; i < a.length; i++) result[i] = a[i] - b[i];
        return result;
    }

    /**
     * Скалярное умножение вектора на число.
     */
    public static float[] multiply(float[] v, float scalar) {
        float[] result = new float[v.length];
        for (int i = 0; i < v.length; i++) result[i] = v[i] * scalar;
        return result;
    }

    /**
     * Вычисление вектора по выражению, заданному списком пар (слово, коэффициент).
     * Пример: выражение "король - мужчина + женщина" задаётся как
     *   List.of(new WordCoeff("король", 1), new WordCoeff("мужчина", -1), new WordCoeff("женщина", 1))
     */
    public static float[] computeVector(Map<String, float[]> wordVectors, List<WordCoeff> expression) {
        if (expression.isEmpty()) throw new IllegalArgumentException("Empty expression");
        int dim = wordVectors.values().iterator().next().length;
        float[] result = new float[dim];
        for (WordCoeff term : expression) {
            float[] vec = wordVectors.get(term.word);
            if (vec == null) throw new IllegalArgumentException("Word not found: " + term.word);
            for (int i = 0; i < dim; i++) result[i] += vec[i] * term.coeff;
        }
        return result;
    }

    // Простой класс для хранения слова и коэффициента
    public static class WordCoeff {
        public final String word;
        public final float coeff;
        public WordCoeff(String word, float coeff) { this.word = word; this.coeff = coeff; }
    }

    /**
     * Поиск ближайших слов к вектору (например, результату векторной арифметики).
     */
    public static List<Map.Entry<String, Double>> findNearestToVector(
            Map<String, float[]> wordVectors,
            float[] targetVec,
            int topK,
            Set<String> excludeWords) {
        List<Map.Entry<String, Double>> similarities = new ArrayList<>();
        for (Map.Entry<String, float[]> entry : wordVectors.entrySet()) {
            if (excludeWords != null && excludeWords.contains(entry.getKey())) continue;
            double sim = cosineSimilarity(targetVec, entry.getValue());
            similarities.add(new AbstractMap.SimpleEntry<>(entry.getKey(), sim));
        }
        similarities.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
        return similarities.stream().limit(topK).collect(Collectors.toList());
    }
}
