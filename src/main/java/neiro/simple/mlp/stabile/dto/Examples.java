package neiro.simple.mlp.stabile.dto;

/**Полный набор учебных примеров для обучения и тестирования
 * @param testData - данные для тестирования
 * @param trainData - данные для обучения*/
public record Examples(Example[] trainData, Example[] testData) {
}
