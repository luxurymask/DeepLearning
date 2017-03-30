package BP;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import util.Utilities;

public class BPNetworkMulti {

	private double[][] layers;
	private double[] targets;
	private double[][] deltas;

	private final int layerNum;
	private final int[] layerSizes;

	/**
	 * 学习率。
	 */
	private final double eta;

	/**
	 * 动量因子。
	 */
	private final double momentum;

	/**
	 * 权值。
	 */
	private List<double[][]> weights = new ArrayList<double[][]>();

	/**
	 * 上一次更新的权值。
	 */
	private List<double[][]> preVWeights = new ArrayList<double[][]>();

	private final Random random;

	public BPNetworkMulti(int[] layerSizes, double eta, double momentum) {
		this.eta = eta;
		this.momentum = momentum;
		this.layerSizes = layerSizes;
		random = new Random(19900216);
		layerNum = layerSizes.length;
		layers = new double[layerNum][];
		deltas = new double[layerNum][];
		for (int i = 0; i < layerNum; i++) {
			int layerSizeNow = layerSizes[i];
			deltas[i] = new double[layerSizeNow];
			layers[i] = new double[layerSizeNow + 1];
			if (i != layerNum - 1) {
				// layers[i] = new double[layerSizeNow + 1];
				int layerSizeNext = layerSizes[i + 1];
				double[][] weight = new double[layerSizeNext][layerSizeNow + 1];
				randomizeWeights(weight);
				weights.add(weight);
				double[][] preWeight = new double[layerSizeNext][layerSizeNow + 1];
				preVWeights.add(preWeight);
			} else {
				// layers[i] = new double[layerSizeNow];
				targets = new double[layerSizeNow];
			}
		}
	}

	public BPNetworkMulti(int[] layerSizes) {
		this(layerSizes, 0.25, 0.9);
	}

	public void train(double[] trainData, double[] target) {
		loadInput(trainData);
		loadTarget(target);
		forward();
		calculateDelta();
		adjustWeights();
	}

	public void showWeights() {
		for (int i = 0; i < weights.size(); i++) {
			double[][] w = weights.get(i);
			for (int j = 0; j < w.length; j++) {
				double[] ww = w[j];
				for (int k = 0; k < ww.length; k++) {
					System.out.println(ww[k]);
				}
			}
		}
	}

	public double[] test(double[] inputData) {
		if (inputData.length != layers[0].length - 1) {
			throw new IllegalArgumentException("Size Do Not Match.");
		}
		System.arraycopy(inputData, 0, layers[0], 0, inputData.length);
		forward();
		return layers[layerNum - 1];
	}

	private void loadTarget(double[] targetData) {
		int targetsSize = targetData.length;
		if (targetsSize != this.targets.length) {
			throw new IllegalArgumentException("Size Does Not Match.");
		}
		System.arraycopy(targetData, 0, this.targets, 0, targetsSize);
	}

	private void loadInput(double[] inputData) {
		int inputSize = inputData.length;
		if (inputSize != layers[0].length - 1) {
			throw new IllegalArgumentException("Size Does Not Match.");
		}
		System.arraycopy(inputData, 0, layers[0], 0, inputSize);
	}

	private void forward() {
		for (int i = 0; i < layerNum - 1; i++) {
			forward(layers[i], layers[i + 1], weights.get(i));
		}
	}

	private void forward(double[] layer0, double[] layer1, double[][] weight) {
		layer0[layer0.length - 1] = 1.0;
		for (int i = 0; i < layer1.length - 1; i++) {
			double sum = 0;
			for (int j = 0; j < layer0.length; j++) {
				sum += weight[i][j] * layer0[j];
			}
			layer1[i] = Utilities.sigmoid(sum);
		}
	}

	private void calculateDelta() {
		outputErr();
		for (int i = layerNum - 2; i > 0; i--) {
			// 计算第i层delta
			hiddenErr(layers[i], deltas[i + 1], deltas[i], weights.get(i));
		}
	}

	/**
	 * 计算输出层delta
	 */
	private void outputErr() {
		double[] outputDelta = deltas[layerNum - 1];
		for (int i = 0; i < outputDelta.length; i++) {
			double output = layers[layerNum - 1][i];
			outputDelta[i] = output * (1d - output) * (targets[i] - output);
		}
	}

	private void hiddenErr(double[] hiddenLayer, double[] outputDelta, double[] hiddenDelta, double[][] weights) {
		for (int i = 0; i < hiddenDelta.length; i++) {
			double o = hiddenLayer[i];
			double sum = 0;
			for (int j = 0; j < outputDelta.length; j++) {
				sum += weights[j][i] * outputDelta[j];
			}
			hiddenDelta[i] = o * (1d - o) * sum;
		}
	}

	private void adjustWeights() {
		for (int i = layerNum - 2; i >= 0; i--) {
			// 调整第i层与第i+1层间的权值。
			adjustWeights(deltas[i + 1], layers[i], i);
		}
	}

	/**
	 * 
	 * @param delta
	 * @param layer
	 * @param layerNum
	 *            调整第layerNum层与第layerNum+1层间的权值。
	 */
	private void adjustWeights(double[] delta, double[] layer, int layerNum) {
		int layerSizeNow = layer.length;
		int layerSizeNext = delta.length;
		double[][] weight = weights.get(layerNum);
		double[][] preV = preVWeights.get(layerNum);
		for (int i = 0; i < delta.length; i++) {
			for (int j = 0; j < layer.length; j++) {
				double newVal = momentum * preVWeights.get(layerNum)[i][j] + eta * delta[i] * layer[j];
				weight[i][j] += newVal;
				preV[i][j] = newVal;
			}
		}
		weights.set(layerNum, weight);
		preVWeights.set(layerNum, preV);
	}

	private void randomizeWeights(double[][] matrix) {
		for (int i = 0, len = matrix.length; i != len; i++) {
			for (int j = 0, len2 = matrix[i].length; j != len2; j++) {
				double real = random.nextDouble();
				matrix[i][j] = random.nextDouble() > 0.5 ? real : -real;
			}
		}
	}
}
