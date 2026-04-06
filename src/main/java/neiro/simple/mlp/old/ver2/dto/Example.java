package neiro.simple.mlp.old.ver2.dto;

/**Класс содержащий в себе один учебный пример. Содержит два вектора
 * @param input - входной вектор
 * @param target - целевой вектор*/
public record Example(double[] input, double[] target) {
}
