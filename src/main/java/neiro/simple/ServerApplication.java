package neiro.simple;

import neiro.simple.mlp.stabile.AsyncNeirons;
import neiro.simple.mlp.stabile.dto.Examples;
import neiro.simple.mlp.stabile.model.Layer;
import neiro.simple.mlp.stabile.model.Net;
import neiro.simple.mlp.stabile.task.TaskMNIST;
import neiro.simple.mlp.stabile.train.IWeightOptimaizer;
import neiro.simple.mlp.stabile.train.Train;
import neiro.simple.mlp.stabile.train.TrainUtil;

import java.util.List;

public class ServerApplication {
	public static void main(String[] args) {
		Examples examplesMNIST = new TaskMNIST().getData(null);

		Net netMnist1 = new Net(List.of(
				(layer) -> new Layer.input(784),
				(layer) -> new Layer.medium.tanh(100, layer),
				(layer) -> new Layer.output.softmaxAndCrossEntity(10, layer)
		));

		//AsyncNeirons.init(1, false); //в один поток без пачек нейронов
		AsyncNeirons.init(-1, true); //в переменное число потоков с пачками
		Train.train(
				netMnist1,
				1,
				examplesMNIST,
				TrainUtil.estimationMax,
				0.1,
				new IWeightOptimaizer.classic(0.02),
				-1,
				5000
		);


		//ExampleMLP.xor();
		//ExampleMLP.mnist();
		//ExampleMLP.iris();
	}
}
