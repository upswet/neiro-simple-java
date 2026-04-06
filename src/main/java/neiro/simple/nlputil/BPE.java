package neiro.simple.nlputil;

import lombok.extern.slf4j.Slf4j;

import java.util.*;
/*
BPE решает ключевую проблему токенизации текста для языковых моделей: баланс между размером словаря и способностью обрабатывать неизвестные слова.

Без BPE возможны два неоптимальных варианта:
    Пословная токенизация — словарь становится слишком большим (сотни тысяч или миллионы слов), много OOV‑токенов (out‑of‑vocabulary — слов, которых нет в словаре).
    Посимвольная токенизация — словарь маленький, но каждое слово разбивается на много токенов, что увеличивает вычислительные затраты.


Как работает BPE
Алгоритм строит словарь подслов (subwords) снизу вверх:
Начинает с базового набора — отдельных символов (или байтов).
Итеративно находит самую частую пару соседних токенов.
Объединяет эту пару в новый токен.
Заменяет все вхождения пары на новый токен.
Повторяет шаги 2–4 заданное число раз (обычно тысячи итераций).

Пример (упрощённый):
Исходный текст: «low lower lowest».
Начальный словарь: {l, o, w, e, r, s, t, _} (символ _ обозначает конец слова).
Шаг 1: самая частая пара (l, o) → объединяем в lo.
Шаг 2: самая частая пара (lo, w) → объединяем в low.
Шаг 3: самая частая пара (e, r) → объединяем в er.
Шаг 4: самая частая пара (e, s) → объединяем в es.

Результат:
low = 1 токен,
lower = low + er,
lowest = low + es + t.


Частотные слова становятся одним токеном, редкие — разбиваются на части. Это сокращает общее число токенов в тексте.

Скорость обработки. Меньше токенов → быстрее инференс.
Потребление памяти. Компактный словарь требует меньше ресурсов.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Вот краткий алгоритм BPE (Byte Pair Encoding) для токенизации:

Инициализация
    Разбить весь корпус текста на символы (или байты).
    Каждому уникальному символу присвоить идентификатор — начальный словарь.

Подсчёт частот пар
    Пройти по всем последовательностям токенов в корпусе.
    Для каждой смежной пары токенов (a, b) увеличить счётчик частоты.

Выбор самой частой пары
    Найти пару (a, b) с максимальной частотой.

Слияние
    Создать новый токен ab, добавить его в словарь.
    Во всех последовательностях заменить каждое вхождение пары (a, b) на новый токен ab.

Повтор
    Повторять шаги 2–4 до тех пор, пока размер словаря не достигнет заданного предела (или не останется пар с частотой > 1).

На выходе получается словарь субсловных токенов, который используется для кодирования и декодирования текста.
*/

@Slf4j
public class BPE {
    public static void main(){
        String text = "Пони тоже кони. Наши маленькие пони. Хорошие пони poni";
        BiMap<String, Integer> dict = new BiMap<String, Integer>();
        List<Integer> list = bpe(dict,text, 999);
        log.info(list.toString());

        List<Integer> l1 = encode(text, dict);
        log.info(l1.toString());
        String s1 = decode(l1, dict);
        log.info(s1);

        list = bpe(dict,text+ " а это новая строка иди-иди ты на рога 56", 999);
        log.info(list.toString());

    }

    /**Текст в список токенов
     * @param dict - словарь
     * @param text  - кодируемый текст
     * @return - список токенов*/
    public static List<Integer> encode(String text, BiMap<String, Integer> dict){
        List<Integer> result = new ArrayList<>();
        List<String> tokens = new ArrayList<>(dict.getKeys());
        tokens.sort(Comparator.comparingInt(String::length).reversed());

        int i = 0;
        while (i < text.length()) {
            boolean matched = false;
            for (String token : tokens) {
                if (text.startsWith(token, i)) {
                    result.add(dict.getByKey(token));
                    i += token.length();
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                // Символа нет в словаре – можно выбросить исключение или добавить спецтокен
                throw new RuntimeException("Unknown character: " + text.charAt(i));
            }
        }
        return result;
    }

    /**Список токенов в текст
     * @param dict - словарь
     * @param list - список токенов
     * @return - расшифрованный текст*/
    public static String decode(List<Integer> list, BiMap<String, Integer> dict){
        StringBuilder sb = new StringBuilder();
        for(Integer token : list)
            sb.append(dict.getByValue(token));
        return sb.toString();
    }

    /**Реализует BPE (получение словаря токенов из текста)
     * @param maxDictSize - максимальный размер словаря
     * @param text - текст на котором составляем словарь
     * @param dict - словарь который расширяем или пустой словарь если заполняем новый
     * @return - закодированный с применением указанного словаря текст в виде последоатлеьного списка токенов*/
    public static List<Integer> bpe(BiMap<String, Integer> dict, String text, int maxDictSize){
        // ----- 1. Инициализация: разбиваем текст на символы -----
        List<Integer> tokenIds = new ArrayList<>();
        int nextTokenId = dict.size() + 1;
        for (char c : text.toCharArray()) {
            String s = String.valueOf(c);
            Integer id = dict.getByKey(s);
            if (id == null) {
                dict.put(s, nextTokenId);
                id = nextTokenId;
                nextTokenId++;
            }
            tokenIds.add(id);
        }

        // ----- 2. Начальные частоты пар -----
        Map<Pair, Integer> freqMap = new HashMap<>();
        for (int i = 0; i < tokenIds.size() - 1; i++) {
            Pair p = new Pair(tokenIds.get(i), tokenIds.get(i + 1));
            freqMap.put(p, freqMap.getOrDefault(p, 0) + 1);
        }

        // ----- 3. Приоритетная очередь (максимальная частота) -----
        PriorityQueue<Pair> pq = new PriorityQueue<>((p1, p2) ->
                Integer.compare(freqMap.getOrDefault(p2, 0), freqMap.getOrDefault(p1, 0))
        );
        pq.addAll(freqMap.keySet());

        // ----- 4. Цикл слияний -----
        while (nextTokenId <= maxDictSize && !pq.isEmpty()) {
            // Берём пару с наибольшей частотой
            Pair best = pq.poll();
            int curFreq = freqMap.getOrDefault(best, 0);
            if (curFreq < 2) break;   // нет пар с частотой >=2

            int a = best.first;
            int b = best.second;
            int newTokenId = nextTokenId++;

            // Создаём строковый токен и добавляем в словарь
            String newTokenStr = dict.getByValue(a) + dict.getByValue(b);
            dict.put(newTokenStr, newTokenId);

            // ----- Слияние: проходим по списку, заменяем (a,b) на newTokenId -----
            List<Integer> newTokenIds = new ArrayList<>(tokenIds.size());
            // Для инкрементального обновления частот запоминаем изменения
            Map<Pair, Integer> freqDelta = new HashMap<>(); // изменения: +1 или -1

            for (int i = 0; i < tokenIds.size(); i++) {
                // Проверяем, образует ли текущий и следующий токены пару (a,b)
                if (i < tokenIds.size() - 1 && tokenIds.get(i) == a && tokenIds.get(i + 1) == b) {
                    // Заменяем (a,b) на newTokenId
                    // Удаляем старую пару (a,b)
                    Pair oldPair = new Pair(a, b);
                    freqDelta.put(oldPair, freqDelta.getOrDefault(oldPair, 0) - 1);

                    // Удаляем пару, которая была слева от (a,b) (если есть)
                    if (i > 0) {
                        Pair leftPair = new Pair(tokenIds.get(i - 1), a);
                        freqDelta.put(leftPair, freqDelta.getOrDefault(leftPair, 0) - 1);
                        // Новая пара (левый, newTokenId)
                        Pair newLeftPair = new Pair(tokenIds.get(i - 1), newTokenId);
                        freqDelta.put(newLeftPair, freqDelta.getOrDefault(newLeftPair, 0) + 1);
                    }
                    // Удаляем пару, которая была справа от (a,b) (если есть)
                    if (i + 2 < tokenIds.size()) {
                        Pair rightPair = new Pair(b, tokenIds.get(i + 2));
                        freqDelta.put(rightPair, freqDelta.getOrDefault(rightPair, 0) - 1);
                        // Новая пара (newTokenId, правый)
                        Pair newRightPair = new Pair(newTokenId, tokenIds.get(i + 2));
                        freqDelta.put(newRightPair, freqDelta.getOrDefault(newRightPair, 0) + 1);
                    }

                    newTokenIds.add(newTokenId);
                    i++; // пропускаем следующий элемент (b)
                } else {
                    newTokenIds.add(tokenIds.get(i));
                }
            }

            // Применяем изменения к freqMap
            for (Map.Entry<Pair, Integer> entry : freqDelta.entrySet()) {
                Pair p = entry.getKey();
                int delta = entry.getValue();
                int newVal = freqMap.getOrDefault(p, 0) + delta;
                if (newVal <= 0) {
                    freqMap.remove(p);
                } else {
                    freqMap.put(p, newVal);
                }
            }

            // Обновляем список токенов
            tokenIds = newTokenIds;

            // Перестраиваем приоритетную очередь на основе актуальной freqMap
            pq.clear();
            pq.addAll(freqMap.keySet());

            // (Опционально) если достигли нужного размера словаря – выход
            if (nextTokenId > maxDictSize) break;
        }

        return tokenIds;
    }
}


final class Pair {
    final int first;
    final int second;
    Pair(int first, int second) {
        this.first = first;
        this.second = second;
    }
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Pair)) return false;
        Pair pair = (Pair) o;
        return first == pair.first && second == pair.second;
    }
    @Override
    public int hashCode() {
        return 31 * first + second;
    }
    @Override
    public String toString() {
        return "(" + first + "," + second + ")";
    }
}