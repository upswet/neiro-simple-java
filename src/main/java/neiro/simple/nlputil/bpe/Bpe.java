package neiro.simple.nlputil.bpe;


import lombok.extern.slf4j.Slf4j;

import java.util.*;

/**Моя релазизация BPE*/
@Slf4j
public class Bpe {
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
        Set<String> tokens = dict.getKeys();
        List<String> list = new ArrayList<>(tokens);
        list.sort((a, b) -> Integer.compare(b.length(), a.length()));
        for(String el : list){
            String sToken = "#START_"+dict.getByKey(el).toString()+"_END#";
            text=text.replace(el, sToken);
        }

        List<Integer> result = new ArrayList<>();
        for(String s : text.split("_END#")){
            String sn = s.substring("#START_".length());
            result.add(Integer.valueOf(sn));
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
     * @return - токенизированный текст*/
    public static List<Integer> bpe(BiMap<String, Integer> dict, String text, int maxDictSize){
        List<Integer> list = new ArrayList<>();
        int tokenNumber = dict.size()+1;

        if (tokenNumber>=maxDictSize)
            throw new RuntimeException("dict size less to max!");

        //Инициализация
        for(char c : text.toCharArray()) {
            String s = String.valueOf(c);
            if (dict.containsKey(s))
                list.add(dict.getByKey(s));
            else {
                list.add(tokenNumber);
                dict.put(s, tokenNumber);
                tokenNumber++;
            }
        }

        //Увеличиваем словарь
        /*todo Идея для ускорения: чтобы заново не сканировать весь список токенов для построения freqMap:
              Хранить частоты пар в Map<Pair, Integer>. record Pair(int first, int second))
              При слиянии пары (a, b) → c удалить пары, затрагивающие a и b, и добавить новые пары (предыдущий токен, c) и (c, следующий токен).
              Поддерживать максимальную частоту с помощью PriorityQueue или TreeMap
         */
        //todo: возможно стоит переделать чтобы создавал пары только в границах слова
        while (tokenNumber < maxDictSize) {
            //Подсчёт частот пар и выбор самой частой пары
            int maxFreq = 1;
            List<Integer> maxKey = null;
            Map<List<Integer>, Integer> freqMap = new HashMap<>();
            for (int i = 1; i < list.size(); i++) {
                List<Integer> key = List.of(list.get(i-1), list.get(i));
                int freq = freqMap.getOrDefault(key, 0) + 1;
                freqMap.put(key, freq);

                if (freq > maxFreq) {
                    maxFreq = freq;
                    maxKey = key;
                }
            }

            if (maxKey==null)
                break; //все частоты равны единице, пожалуй хватит увеличивать словарь

            //Слияние
            //Создать новый токен ab, добавить его в словарь.
            String newKey = dict.getByValue(maxKey.get(0)) + dict.getByValue(maxKey.get(1));
            dict.put(newKey, tokenNumber);
            //Во всех последовательностях заменить каждое вхождение пары (a, b) на новый токен ab.
            for (int i = 1; i < list.size() ; i++){
                List<Integer> key = List.of(list.get(i-1), list.get(i));
                if (key.equals(maxKey)) {
                    list.set(i-1, tokenNumber);
                    list.remove(i);
                    i--;
                }
            }
            tokenNumber++;
        }

        return list;
    }
}
