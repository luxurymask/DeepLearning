package perceptron;

import java.util.Random;

import util.Utilities;

public class SimplePerceptron {
	public Layer[] weights;
	public Layer bias;
	public LayerFactory layerFactory;

	Random rng;
	static Long randomSeed;

	/**
	 * 构造方法
	 * @param numInput
	 * @param numOutput
	 */
	public SimplePerceptron(int numInput, int numOutput){
		rng = new Random();
		if(randomSeed != null){
			rng.setSeed(randomSeed);
		}
		layerFactory = new LayerFactory();

		bias = layerFactory.create(numOutput);
		weights = new Layer[numOutput];
		for(int i = 0;i < numOutput;i++){
			bias.set(i, new Float(rng.nextGaussian()));
			weights[i] = layerFactory.create(numInput);
			for(int j = 0;j < numInput;j++){
				weights[i].set(j, new Float(rng.nextGaussian()));
			}
		}
	}

	public Layer compute(final Layer input){
		Layer output = layerFactory.create(bias.size());
		for(int i = 0;i < weights.length;i++){
			for(int j = 0;j < input.size();j++){
				output.add(i, weights[i].get(j) * input.get(j));
			}
			output.set(i, Utilities.step(output.get(i) + bias.get(i)));
		}
		return output;
	}
}
