package perceptron;

public class ANDPerceptron {
	final SimplePerceptron perceptron;
	final SimplePerceptronTrainer trainer;
	final LayerFactory layerFactory = new LayerFactory();
	final float step;
	final Layer[] inputBatch;
	final Layer[] outputFlagBatch;

	public ANDPerceptron(){
		perceptron = new SimplePerceptron(2, 1);
		step = 0.1f;
		trainer = new SimplePerceptronTrainer(layerFactory, step);
		float[] input1 = {0, 0};
		Layer layer1 = new Layer(input1);
		float[] input2 = {0, 1};
		Layer layer2 = new Layer(input2);
		float[] input3 = {1, 0};
		Layer layer3 = new Layer(input3);
		float[] input4 = {1, 1};
		Layer layer4 = new Layer(input4);
		inputBatch = new Layer[4];
		inputBatch[0] = layer1;
		inputBatch[1] = layer2;
		inputBatch[2] = layer3;
		inputBatch[3] = layer4;


		float[] input11 = {0};
		Layer layer11 = new Layer(input11);
		float[] input22 = {0};
		Layer layer22 = new Layer(input22);
		float[] input33 = {0};
		Layer layer33 = new Layer(input33);
		float[] input44 = {1};
		Layer layer44 = new Layer(input44);
		outputFlagBatch = new Layer[4];
		outputFlagBatch[0] = layer11;
		outputFlagBatch[1] = layer22;
		outputFlagBatch[2] = layer33;
		outputFlagBatch[3] = layer44;
	}

	public static void main(String[] args){
		ANDPerceptron myPerceptron = new ANDPerceptron();
		myPerceptron.trainer.learn(myPerceptron.perceptron, myPerceptron.inputBatch, myPerceptron.outputFlagBatch);
		myPerceptron.trainer.test(myPerceptron.perceptron, myPerceptron.inputBatch, myPerceptron.outputFlagBatch);
	}
}
