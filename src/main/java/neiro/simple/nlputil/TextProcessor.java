package neiro.simple.nlputil;

import lombok.SneakyThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Stream;

public class TextProcessor {
    /**
     * Обрабатывает текстовый файл и возвращает массив слов.
     *
     * @param filePath путь к файлу
     * @param minLength - минимально допустимая длинна
     * @return массив слов
     */
    @SneakyThrows
    public static String[] processFile(String filePath, int minLength, String regex)  {
        Path path = Paths.get(filePath);

        try (Stream<String> lines = Files.lines(path)) {
            return lines
                    // Убираем пустые строки (и строки только из пробелов)
                    .filter(line -> !line.trim().isEmpty())
                    // Приводим к нижнему регистру
                    .map(String::toLowerCase)
                    // Заменяем знаки препинания на пробелы
                    .map(line -> line.replaceAll("\\p{Punct}", " "))
                    // Удаляем лишние пробелы по краям
                    .map(String::trim)
                    // Применяем регулярку оставляя только нужные символы
                    .map(line -> line.replaceAll(regex, " "))
                    // Игнорируем строки, которые стали пустыми после замены знаков
                    .filter(line -> !line.isEmpty())
                    // Разбиваем каждую строку на слова (по любым пробельным символам)
                    .flatMap(line -> Arrays.stream(line.split("\\s+")))
                    // Исключаем возможные пустые слова
                    .filter(word -> !word.isEmpty())
                    // Теперь фильтруем отдельные слова по длине
                    .filter(word -> word.length() >= minLength)
                    // Собираем результат в массив строк
                    .toArray(String[]::new);
        }
    }

    /**Убрать редкие слова из массива words с частотой меньше чем minWordFreq*/
    public static String[] removeRare(String[] words, int minWordFreq){
        Map<String, Integer> freq = new HashMap<>();
        for (String w : words) {
            freq.put(w, freq.getOrDefault(w, 0) + 1);
        }
        List<String> filtered = new ArrayList<>(words.length);
        for (String w : words) {
            if (freq.get(w) >= minWordFreq) {
                filtered.add(w);
            }
        }
        return filtered.toArray(new String[0]);
    }
}
