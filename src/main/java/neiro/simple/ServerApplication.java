package neiro.simple;

import lombok.SneakyThrows;
import neiro.simple.mlp.*;
import neiro.simple.nlp.TextUtils;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

@SpringBootApplication
public class ServerApplication {
	public static void main(String[] args) {
		//SpringApplication.run(ServerApplication.class, args);

		//ExampleMLP.xor();
		//ExampleMLP.mnist();
		ExampleMLP.iris();

		/*	Set<String> set = TextUtils.processText("d:\\Work\\Project\\neiro\\txt");
		TextUtils.saveVocabularyToFile(set, "d:\\Work\\Project\\neiro\\worlds.file");
		set = TextUtils.loadVocabularyFromFile("d:\\Work\\Project\\neiro\\worlds.file");
		List<String> list = new ArrayList<>(set);
		double[][] embedings = VocToEmbed.vocToEmbeding(list, 10);
		System.out.println("is end");
*/

	/*	Net net = new Net(List.of(
				(layer) -> new LayerInput(10, 0.0),
				(layer) -> new LayerMedium.LayerMediumTanh(layer, 4, 0.0),
				(layer) -> new LayerOutput.LayerOutputSigmoid(layer,10)
		));

		net.trains(
				(Integer i) -> generateVec(i, 10),
				(Integer i) -> generateVec(i, 10),
				10,
				0.2,
				9000,
				1000,
				new Net.TestWrapper(
						(Integer i) -> generateVec(i, 10),
						(Integer i) -> generateVec(i, 10),
						10,
						estimationMax,
						0.1),
				0.95
		);*/
	}
}
