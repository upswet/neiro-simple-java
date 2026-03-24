package neiro.simple;

import lombok.extern.slf4j.Slf4j;
import neiro.simple.mlp.stabile.AsyncNeirons;
import neiro.simple.mlp.stabile.Utils;
import neiro.simple.mlp.stabile.dto.Examples;
import neiro.simple.mlp.stabile.model.Layer;
import neiro.simple.mlp.stabile.model.Net;
import neiro.simple.mlp.stabile.task.TaskMNIST;
import neiro.simple.mlp.stabile.train.IWeightOptimaizer;
import neiro.simple.mlp.stabile.train.Train;
import neiro.simple.mlp.stabile.train.TrainUtil;

import java.util.List;

@Slf4j
public class ServerApplication {
	public static void main(String[] args) {
		Examples examplesMNIST = new TaskMNIST().getData(null);

		Net netMnist1 = new Net(List.of(
				//input
				(layer) -> new Layer.input(784),
				//medium
				(layer) -> new Layer.medium.tanh(100, layer),
				//(layer) -> new Layer.medium.relu(100, layer),
				//output
				(layer) -> new Layer.output.softmaxAndCrossEntity(10, layer)
				//(layer) -> new Layer.output.sigmoid(10,layer)
				//(layer) -> new Layer.output.relu(10,layer)
		));

		//AsyncNeirons.init(1, false); //в один поток без пачек нейронов
		AsyncNeirons.init(-1, true); //в переменное число потоков с пачками
		Train.train(
				netMnist1,
				1,
				examplesMNIST,
				TrainUtil.estimationMax,
				0.1,
				//new IWeightOptimaizer.classic(0.02),
				new IWeightOptimaizer.adam(0.01),
				100,
				5000
		);


		/*
		//проверка сохраннения / загрузки
		Utils.save("net1", netMnist1);
		Net netLoad = (Net) Utils.load("net1");
		log.info("test={}", Train.test(netLoad, examplesMNIST.testData(), TrainUtil.estimationMax, 0.01));
		 */

		//ExampleMLP.xor();
		//ExampleMLP.mnist();
		//ExampleMLP.iris();
	}
}
