package perceptron;

import java.awt.Canvas;
import java.awt.Color;
import java.awt.Graphics;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import javax.swing.JFrame;

public class ClassifyPerceptron extends Canvas{
	final SimplePerceptron perceptron;
	final SimplePerceptronTrainer trainer;
	final LayerFactory layerFactory = new LayerFactory();
	final float step;
	final static int NUM_TESTBATCH = 101;
	final Layer[] inputBatch = new Layer[100];
	final Layer[] outputFlagBatch = new Layer[100];
	final Layer[] testBatch = new Layer[NUM_TESTBATCH];
	final Layer[] testFlagBatch = new Layer[NUM_TESTBATCH];
	private Layer[] outputBatch = new Layer[NUM_TESTBATCH];

	public ClassifyPerceptron(String inputPath, String outputFlagPath, String testPath, String testFlagPath){
		perceptron = new SimplePerceptron(2, 1);
		step = 0.1f;
		trainer = new SimplePerceptronTrainer(layerFactory, step);

		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(inputPath));
			String lineString = null;
			int line = 0;
			while((lineString = inputReader.readLine()) != null){
				String[] inputStrings = lineString.split(" ");
				int num = inputStrings.length;
				float[] inputLayer = new float[num];
				for(int i = 0;i < num;i++){
					inputLayer[i] = Float.parseFloat(inputStrings[i]);
				}
				inputBatch[line++] = new Layer(inputLayer);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		BufferedReader outputFlagReader = null;
		try{
			outputFlagReader = new BufferedReader(new FileReader(outputFlagPath));
			String lineString = null;
			int line = 0;
			while((lineString = outputFlagReader.readLine()) != null){
				String[] outputFlagStrings = lineString.split(" ");
				int num = outputFlagStrings.length;
				float[] outputFlagLayer = new float[num];
				for(int i = 0;i < num;i++){
					outputFlagLayer[i] = Float.parseFloat(outputFlagStrings[i]);
				}
				outputFlagBatch[line++] = new Layer(outputFlagLayer);
			}
		}catch(IOException e){
			e.printStackTrace();
		}

		BufferedReader testReader = null;
		try{
			testReader = new BufferedReader(new FileReader(testPath));
			String lineString = null;
			int line = 0;
			while((lineString = testReader.readLine()) != null){
				String[] testStrings = lineString.split(" ");
				int num = testStrings.length;
				float[] testLayer = new float[num];
				for(int i = 0;i < num;i++){
					testLayer[i] = Float.parseFloat(testStrings[i]);
				}
				testBatch[line++] = new Layer(testLayer);
			}
		}catch(IOException e){
			e.printStackTrace();
		}

		BufferedReader testFlagReader = null;
		try{
			testFlagReader = new BufferedReader(new FileReader(testFlagPath));
			String lineString = null;
			int line = 0;
			while((lineString = testFlagReader.readLine()) != null){
				String[] testFlagStrings = lineString.split(" ");
				int num = testFlagStrings.length;
				float[] testFlagLayer = new float[num];
				for(int i = 0;i < num;i++){
					testFlagLayer[i] = Float.parseFloat(testFlagStrings[i]);
				}
				testFlagBatch[line++] = new Layer(testFlagLayer);
			}
		}catch(IOException e){
			e.printStackTrace();
		}
	}

	public void paint(Graphics g){
		super.paint(g);
		g.drawLine(50, 50, 50, 718);
		g.drawLine(50, 718, 974, 718);
		Layer[] inputBatch = this.testBatch;
		Layer[] outputBatch = this.outputBatch;
		for(int i = 0;i < inputBatch.length;i++){
			Layer inputLayer = inputBatch[i];
			Layer outputLayer = outputBatch[i];
			if(outputLayer.get(0) == 0.0){
				g.setColor(Color.BLUE);
				g.fillOval(((int)inputLayer.get(0)) * 7 + 50, 768 - (((int)inputLayer.get(1)) * 5 + 50), 10, 10);
			}else if(outputLayer.get(0) == 1.0){
				g.setColor(Color.RED);
				g.fillOval(((int)inputLayer.get(0)) * 7 + 50, 768 - (((int)inputLayer.get(1)) * 5 + 50), 10, 10);
			}
		}
		g.setColor(Color.DARK_GRAY);
		g.drawLine(50, 718 + (int)(this.perceptron.bias.get(0)/this.perceptron.weights[0].get(1) * 5), -(int)(this.perceptron.bias.get(0)/this.perceptron.weights[0].get(0) * 7) + 50, 718);
	}

	public static void main(String[] args){
		String inputPath = "/Users/liuxl/Root/DeepLearning/Perceptron/data/input.dat";
		String outputFlagPath = "/Users/liuxl/Root/DeepLearning/Perceptron/data/outputFlag.dat";
		String testPath = "/Users/liuxl/Root/DeepLearning/Perceptron/data/test.dat";
		String testFlagPath = "/Users/liuxl/Root/DeepLearning/Perceptron/data/testFlag.dat";

		ClassifyPerceptron myPerceptron = new ClassifyPerceptron(inputPath, outputFlagPath, testPath, testFlagPath);
		myPerceptron.trainer.learn(myPerceptron.perceptron, myPerceptron.inputBatch, myPerceptron.outputFlagBatch);
		myPerceptron.outputBatch = myPerceptron.trainer.test(myPerceptron.perceptron, myPerceptron.testBatch, myPerceptron.testFlagBatch);

		Layer[] W = new Layer[2];
		Layer bias = new Layer(1);
		W = myPerceptron.perceptron.weights;
		bias = myPerceptron.perceptron.bias;

		JFrame frame = new JFrame("Perceptron using in coordinate classifying.");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        myPerceptron.setSize(1024, 768);
        frame.add(myPerceptron);

        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
	}
}
