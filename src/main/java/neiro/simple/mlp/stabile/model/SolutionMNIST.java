package neiro.simple.mlp.stabile.model;

import lombok.extern.slf4j.Slf4j;
import neiro.simple.mlp.stabile.dto.Examples;
import neiro.simple.mlp.stabile.dto.TrainResult;
import neiro.simple.mlp.stabile.task.TaskMNIST;
import neiro.simple.mlp.stabile.train.IWeightOptimaizer;
import neiro.simple.mlp.stabile.train.Train;
import neiro.simple.mlp.stabile.train.TrainUtil;

import java.util.List;

/**Решение задачи МНИСТ данной нейросетью*/
@Slf4j
public class SolutionMNIST {

	public static void main(String[] args) {
		Examples examplesMNIST = new TaskMNIST().getData(null);

		/*
		train("RELU", new Net(List.of(
				//input
				(layer) -> new Layer.input(784),
				//medium
				(layer) -> new Layer.medium.relu(128, layer),
				//output
				(layer) -> new Layer.output.relu(10, layer)
		)), examplesMNIST, new IWeightOptimaizer.classic(0.001), 32);

		Релу не самый лучший выбор. Пришлось сделать больше нейронов в промежуточном слое (или больше слоёв), а также меньше коэф обучения и добавить батчи

		15:06:22.938 [main] INFO neiro.simple.mlp.stabile.train.Train -- TrainResult{duration=51875, epoch=1, quality=0.4767}
		15:07:12.590 [main] INFO neiro.simple.mlp.stabile.train.Train -- TrainResult{duration=46567, epoch=2, quality=0.5397}
		15:08:03.172 [main] INFO neiro.simple.mlp.stabile.train.Train -- TrainResult{duration=46896, epoch=3, quality=0.5935}
		15:08:54.459 [main] INFO neiro.simple.mlp.stabile.train.Train -- TrainResult{duration=47133, epoch=4, quality=0.6919}
		15:09:47.983 [main] INFO neiro.simple.mlp.stabile.train.Train -- TrainResult{duration=49782, epoch=5, quality=0.7629}
		15:10:32.272 [main] INFO neiro.simple.mlp.stabile.train.Train -- TrainResult{duration=41018, epoch=6, quality=0.7787}
		 */

		/*тоже самое с хард - немного лучше
		train("HARD", new Net(List.of(
				//input
				(layer) -> new Layer.input(784),
				//medium
				(layer) -> new Layer.medium.hard(128, layer),
				//output
				(layer) -> new Layer.output.hard(10, layer)
		)), examplesMNIST, new IWeightOptimaizer.classic(0.001), 32);

		15:19:28.286 [main] INFO neiro.simple.mlp.stabile.train.Train -- TrainResult{duration=41887, epoch=1, quality=0.7016}
		15:20:16.923 [main] INFO neiro.simple.mlp.stabile.train.Train -- TrainResult{duration=45082, epoch=2, quality=0.7686}
		15:21:07.294 [main] INFO neiro.simple.mlp.stabile.train.Train -- TrainResult{duration=46633, epoch=3, quality=0.8017}
		15:22:00.360 [main] INFO neiro.simple.mlp.stabile.train.Train -- TrainResult{duration=49300, epoch=4, quality=0.8179}
		15:22:53.304 [main] INFO neiro.simple.mlp.stabile.train.Train -- TrainResult{duration=49033, epoch=5, quality=0.8294}
		*/

		/*тангес и софтмакс с адам-оптимизатором и без батчей
		train("tanh+softmax", new Net(List.of(
				//input
				(layer) -> new Layer.input(784),
				//medium
				(layer) -> new Layer.medium.tanh(100, layer),
				//output
				(layer) -> new Layer.output.softmaxAndCrossEntity(10, layer)
		)), examplesMNIST, new IWeightOptimaizer.adam(0.01), -1);
		15:28:10.883 [main] INFO neiro.simple.mlp.stabile.model.SolutionMNIST -- TrainResult{duration=86982, epoch=1, quality=0.9022}
		 */

		//тоже самое но классический оптимизатор
		train("tanh+softmax", new Net(List.of(
				//input
				(layer) -> new Layer.input(784),
				//medium
				(layer) -> new Layer.medium.tanh(100, layer),
				//output
				(layer) -> new Layer.output.softmaxAndCrossEntity(10, layer)
		)), examplesMNIST, new IWeightOptimaizer.classic(0.02), -1);
		// [main] INFO neiro.simple.mlp.stabile.model.SolutionMNIST -- TrainResult{duration=27206, epoch=1, quality=0.9547}
	}

	public static void train(String desc, INetMLP net, Examples examplesMNIST, IWeightOptimaizer optimaizer, Integer batch) {
		log.info("{}", desc);
		TrainResult result = Train.train(
				net,
				1,
				examplesMNIST,
				TrainUtil.estimationMax,
				0.1,
				optimaizer,
				batch,
				-1 //5000
		);
		log.info("{} \n", result.toString());
	}
}
