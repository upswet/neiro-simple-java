package neiro.simple.nlp;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**Служебные утилиты подготовки текста*/
@Slf4j
public class TextUtils {

    // Метод для чтения всех текстовых файлов из директории
    public static String readAllFilesFromDirectory(String directoryPath) throws IOException {
        log.info("starting");
        StringBuilder allText = new StringBuilder();
        Path dirPath = Paths.get(directoryPath);

        try (DirectoryStream<Path> stream = Files.newDirectoryStream(dirPath, "*.txt")) {
            for (Path filePath : stream) {
                if (Files.isRegularFile(filePath)) {
                    log.info("reading file {}", filePath);
                    allText.append(Files.readString(filePath)).append(" ");
                }
            }
        }
        log.info("allText size is {}", allText.length());
        return allText.toString().toLowerCase().replaceAll("[^a-zA-Zа-яА-ЯёЁ0-9\\s]", " ");
    }

    // Метод для разбиения текста на токены (слова)
    public static List<String> tokenize(String text) {
        log.info("starting");
        List<String> tokens = new ArrayList<>();
        String[] words = text.split("\\s+");

        for (String word : words) {
            if (!word.trim().isEmpty()) {
                tokens.add(word.trim());
            }
        }
        log.info("tokens size is {}", tokens.size());
        return tokens;
    }

    // Упрощенная лемматизация (базовые правила для русского языка)
    public static String lemmatizeWord(String word) {
        if (word.length() < 3) return word;

        // Базовые правила для упрощенной лемматизации
        Map<String, String> endings = new HashMap<>();

        // Существительные (окончания)
        endings.put("ами", "");
        endings.put("ями", "");
        endings.put("ах", "");
        endings.put("ях", "");
        endings.put("ей", "");
        endings.put("ом", "");
        endings.put("ем", "");
        endings.put("ов", "");
        endings.put("ев", "");
        endings.put("ий", "ий");
        endings.put("ый", "ый");
        endings.put("ая", "ая");
        endings.put("яя", "яя");
        endings.put("ое", "ое");
        endings.put("ее", "ее");

        // Глаголы
        endings.put("ться", "ть");
        endings.put("тся", "ть");
        endings.put("лся", "ть");
        endings.put("лась", "ть");
        endings.put("лось", "ть");
        endings.put("лись", "ть");

        for (Map.Entry<String, String> entry : endings.entrySet()) {
            String ending = entry.getKey();
            String replacement = entry.getValue();

            if (word.endsWith(ending)) {
                int endIndex = word.length() - ending.length();
                String stem = word.substring(0, endIndex);
                return stem + replacement;
            }
        }

        return word;
    }

    // Упрощенный стемминг (алгоритм Портера в минимальной реализации)
    public static String stemWord(String word) {
        if (word.length() < 3) return word;

        // Шаг 1: удаление окончаний
        String[] step1Endings = {"ами", "ями", "ах", "ях", "ей", "ой", "ом",
                "ем", "ов", "ев", "ий", "ый", "ая", "яя",
                "ое", "ее", "ть", "ти", "л", "ла", "ло", "ли"};

        for (String ending : step1Endings) {
            if (word.endsWith(ending) && word.length() > ending.length() + 2) {
                return word.substring(0, word.length() - ending.length());
            }
        }

        // Шаг 2: удаление суффиксов
        String[] step2Suffixes = {"ость", "ост", "ейш", "ейш"};

        for (String suffix : step2Suffixes) {
            if (word.endsWith(suffix) && word.length() > suffix.length() + 3) {
                return word.substring(0, word.length() - suffix.length());
            }
        }

        return word;
    }

    // Дополнительный метод: удаление стоп-слов
    public static List<String> removeStopWords(List<String> tokens) {
        Set<String> stopWords = new HashSet<>(Arrays.asList(
                "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как",
                "а", "то", "все", "она", "так", "его", "но", "да", "ты", "к",
                "у", "же", "вы", "за", "бы", "по", "только", "ее", "мне", "было",
                "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь",
                "когда", "даже", "ну", "вдруг", "ли", "если", "уже", "или",
                "ни", "быть", "был", "него", "до", "вас", "нибудь", "опять",
                "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
                "может", "они", "тут", "где", "есть", "надо", "ней", "для",
                "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "без",
                "будто", "чего", "раз", "тоже", "себе", "под", "будет", "ж",
                "тогда", "кто", "этот", "того", "потому", "этого", "какой",
                "совсем", "ним", "здесь", "этом", "один", "почти", "мой",
                "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем",
                "всех", "никогда", "можно", "при", "наконец", "два", "об",
                "другой", "хоть", "после", "над", "больше", "тот", "через",
                "эти", "нас", "про", "всего", "них", "какая", "много", "разве",
                "три", "эту", "моя", "впрочем", "хорошо", "свою", "этой",
                "перед", "иногда", "лучше", "чуть", "том", "нельзя", "такой",
                "им", "более", "всегда", "конечно", "всю", "между"
        ));

        List<String> filteredTokens = new ArrayList<>();
        for (String token : tokens) {
            if (!stopWords.contains(token) && token.length() > 2) {
                filteredTokens.add(token);
            }
        }
        return filteredTokens;
    }

    /** Основной метод обработки
     *
     * @param directoryPath - путь к директории с текстовыми файлами
     * @return - множество уникальных слов
     */
    @SneakyThrows
    public static Set<String> processText(String directoryPath) {
        // 1. Чтение файлов
        // 2. Приведение к нижнему регистру
        // 3. Удаление знаков препинания
        String text = readAllFilesFromDirectory(directoryPath);

        // 4. Токенизация
        List<String> tokens = tokenize(text);

        // 5. Лемматизация
        log.info("starting lemmatized");
        List<String> lemmatizedTokens = new ArrayList<>();
        for (String token : tokens) {
            lemmatizedTokens.add(lemmatizeWord(token));
        }

        // 6. Стемминг
        log.info("starting stemmed");
        List<String> stemmedTokens = new ArrayList<>();
        for (String token : lemmatizedTokens) {
            stemmedTokens.add(stemWord(token));
        }

        // 7. Удаление стоп-слов
        log.info("starting removeStopWords");
        List<String> finalTokens = removeStopWords(stemmedTokens);

        // 8. Создание словаря уникальных слов
        return new HashSet<>(finalTokens);
    }

    // Дополнительный метод для сохранения словаря в файл
    @SneakyThrows
    public static void saveVocabularyToFile(Set<String> vocabulary, String filename) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            List<String> sortedWords = new ArrayList<>(vocabulary);
            Collections.sort(sortedWords);

            for (String word : sortedWords) {
                writer.write(word);
                writer.newLine();
            }
        }
    }

    /**
     * Загружает словарь из файла
     *
     * @param filePath путь к файлу со словарем (одно слово на строку)
     * @return Set уникальных слов
     */
    @SneakyThrows
    public static Set<String> loadVocabularyFromFile(String filePath) {
        Set<String> vocabulary = new HashSet<>();
        Path path = Paths.get(filePath);

        // Проверяем существование файла
        if (!Files.exists(path)) {
            System.err.println("File not found: " + filePath);
            return vocabulary; // возвращаем пустой словарь
        }

        try (BufferedReader reader = Files.newBufferedReader(path)) {
            String line;
            while ((line = reader.readLine()) != null) {
                // Убираем лишние пробелы и проверяем, что строка не пустая
                String word = line.trim();
                if (!word.isEmpty()) {
                    vocabulary.add(word);
                }
            }
        }

        System.out.println("Loading " + vocabulary.size() + " worlds from file: " + filePath);
        return vocabulary;
    }
}
