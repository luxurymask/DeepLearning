package perceptron;

public class SimplePerceptronTrainer {
	private LayerFactory layerFactory;
	private float step;

	public SimplePerceptronTrainer(LayerFactory layerFactory, float step){
		this.layerFactory = layerFactory;
		this.step = step;
	}

	public void learn(final SimplePerceptron perceptron, Layer[] inputBatch, Layer[] outputFlagBatch){
		int count = 0;
		int batchSize = inputBatch.length;
		boolean flagBatch = false;
		while(!flagBatch){
			count++;
			System.out.println("Iteration " + count + " starts...");
			for(int i = 0;i < batchSize;i++){
				flagBatch = true;
				Layer input = inputBatch[i];
				Layer outputFlag = outputFlagBatch[i];
				Layer output = perceptron.compute(input);
				boolean flagOutput = false;
				for(int j = 0;j < output.size() && flagOutput == false;j++){
					float error = outputFlag.get(j) - output.get(j);
					if(error != 0.0f){
						for(int k = 0;k < input.size();k++){
							perceptron.weights[j].add(k, input.get(k) * error * step);
						}
						perceptron.bias.add(j, error * step);
						flagBatch = false;
						break;
					}
					if(j == output.size() - 1){
						flagOutput = true;
					}
				}
				if(flagBatch == false){
					break;
				}
			}
		}
	}

	public Layer[] test(final SimplePerceptron perceptron, Layer[] inputBatch, Layer[] outputFlagBatch){
		int inputSize = inputBatch.length;
		Layer[] outputBatch = new Layer[inputSize];
		for(int i = 0;i < inputSize;i++){
			System.out.print("Testing input " + i + "...");
			Layer input = inputBatch[i];
			Layer outputFlag = outputFlagBatch[i];
			Layer output = perceptron.compute(input);
			outputBatch[i] = output;
			for(int j = 0;j < output.size();j++){
				System.out.print("Output neuron " + j + " comes: " + output.get(j) + ", which is ");
				System.out.print((output.get(j) - outputFlag.get(j)) == 0.0 ? "correct " : "wrong ");
			}
			System.out.println("");
		}
		return outputBatch;
	}
}
