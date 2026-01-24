package neiro.simple;

import lombok.SneakyThrows;
import neiro.simple.obj.*;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

@SpringBootApplication
public class ServerApplication {
	public static void main(String[] args) {
		//SpringApplication.run(ServerApplication.class, args);

		//XOR

		Net net = new Net(List.of(
				(layer) -> new LayerInput(2),
				(layer) -> new LayerMedium.LayerMediumRelu(layer, 3),
				(layer) -> new LayerOutput.LayerOutputRelu(layer,1)
		));

		Net.save("net1.save", net);
		Net net2=Net.load("net1.save");
		net.print();

		net.trains(
				new double[][]{
						new double[]{0F, 0F},
						new double[]{1F, 0F},
						new double[]{0F, 1F},
						new double[]{1F, 1F},
				},
				new double[][]{
						new double[]{0F},
						new double[]{1F},
						new double[]{1F},
						new double[]{0F},
				},
				0.2,
				5000,
				100,
				()->net.tests(
						new double[][]{
								new double[]{0F, 0F},
								new double[]{1F, 0F},
								new double[]{0F, 1F},
								new double[]{1F, 1F},
						},
						new double[][]{
								new double[]{0F},
								new double[]{1F},
								new double[]{1F},
								new double[]{0F},
						},
						estimationXor,
						0.1),
				0.98
		);


		System.out.println("{1,1} = "+ Arrays.toString(net.forward((new double[]{1, 1}))));
		System.out.println("{1,0} = "+ Arrays.toString(net.forward((new double[]{1, 0}))));
		System.out.println("{0,1} = "+ Arrays.toString(net.forward((new double[]{0, 1}))));
		System.out.println("{0,0} = "+ Arrays.toString(net.forward((new double[]{0, 0}))));


		//MNIST
	/*	List<double[]> datasTrain = new ArrayList<>();
		List<double[]> targetsTrain = new ArrayList<>();
		List<double[]> datasTest = new ArrayList<>();
		List<double[]> targetsTest = new ArrayList<>();

		prepareDataForMnist(targetsTrain, datasTrain, "d:\\Work\\Project\\0files\\mnist\\mnist_train.csv");
		prepareDataForMnist(targetsTest, datasTest, "d:\\Work\\Project\\0files\\mnist\\mnist_test.csv");

		Net net = new Net(List.of(
				(layer) -> new LayerInput(784),
				(layer) -> new LayerMedium.LayerMediumSigmoid(layer, 100),
				(layer) -> new LayerOutput.LayerOutputSigmoid(layer,10)
		));

		net.trains(
				datasTrain.toArray(double[][]::new),
				targetsTrain.toArray(double[][]::new),
				0.2,
				1,
				-1,
				()->net.tests(
						datasTest.toArray(double[][]::new),
						targetsTest.toArray(double[][]::new),
						estimation,
						0.01),
				0.98
		);
	 */
	}

	/**Подготовка данных MNIST*/
	@SneakyThrows
	private static void prepareDataForMnist(List<double[]> targets, List<double[]> datas, String fileName) {
		List<String> list = Files.readAllLines(Paths.get(fileName));
		for (String s : list) {
			String[] sarr = s.split(",");

			double[] target = new double[10];
			Arrays.fill(target, 0.01);
			target[Integer.valueOf(sarr[0])] = 0.99;
			targets.add(target);

			double[] data = new double[sarr.length - 1];
			for (int j = 1; j < sarr.length; j++)
				data[j - 1] = (Double.valueOf(sarr[j]) / 255.0) * 0.99 + 0.01;

			datas.add(data);
		}
	}

	/**вернёт индекс максимального элемента из вектора*/
	public static Integer findMax(double[] arr) {
		double max = -999F;
		Integer imax = -1;
		for (int i = 0; i < arr.length; i++)
			if (arr[i] > max) {
				max = arr[i];
				imax = i;
			}
		return imax;
	}

	public static Net.Estimation estimation = (double[] outputVec, double[] expectedVec, double acceptableError) ->{
		return findMax(outputVec).equals(findMax(expectedVec));
	};

	public static Net.Estimation estimationXor = (double[] outputVec, double[] expectedVec, double acceptableError) ->{
		double loss=0;
		for(int i=0; i<outputVec.length;i++)
			loss = loss + Math.abs(expectedVec[i]-outputVec[i]);
		return loss < acceptableError;
	};
}
