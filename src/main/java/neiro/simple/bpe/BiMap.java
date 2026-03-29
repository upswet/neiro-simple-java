package neiro.simple.bpe;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class BiMap <K, V> {
    private final Map<K, V> keyToValue = new HashMap<>();
    private final Map<V, K> valueToKey = new HashMap<>();

    public void put(K key, V value) {
        keyToValue.put(key, value);
        valueToKey.put(value, key);
    }

    public V getByKey(K key) {
        return keyToValue.get(key);
    }

    public V getByKeyOrDefault(K key, V defaultValue) {
        return keyToValue.getOrDefault(key, defaultValue);
    }

    public K getByValue(V value) {
        return valueToKey.get(value);
    }

    public K getByValueOrDefault(V value, K defaultValue) {
        return valueToKey.getOrDefault(value, defaultValue);
    }

    public boolean containsKey(K key) {
        return keyToValue.containsKey(key);
    }

    public boolean containsValue(V value) {
        return valueToKey.containsKey(value);
    }

    public int size(){
        return keyToValue.size();
    }

    public Set<K> getKeys(){
        return keyToValue.keySet();
    }

    public Set<V> getValues(){
        return valueToKey.keySet();
    }

}