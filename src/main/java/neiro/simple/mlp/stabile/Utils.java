package neiro.simple.mlp.stabile;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

/**Утилиты для работы с нейросетью*/
public class Utils {
    /**
     * Сохранить обученную нейросеть в файл по его пути
     *
     * @param filePath - путь к файлу в который сохраняем
     * @param object   - сохраняемый объект
     */
    public static void save(String filePath, Object object) {
        try {
            var fileOutput = new FileOutputStream(filePath);
            var objectOutput = new ObjectOutputStream(fileOutput);
            objectOutput.writeObject(object);
            fileOutput.flush();
            objectOutput.flush();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        System.out.println("saving in " + filePath);
    }

    /**
     * Загрузить обученную нейросеть из файла
     *
     * @param filePath - путь к файлу из которого загружаем нейронку
     * @return - нейросеть
     */
    public static  Object load(String filePath) {
        try {
            var fileInput = new FileInputStream(filePath);
            var objectInput = new ObjectInputStream(fileInput);
            Object object = objectInput.readObject();
            fileInput.close();
            objectInput.close();
            System.out.println("loading from " + filePath);
            return (Object)object;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
