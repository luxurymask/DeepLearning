package RBM;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;

public class RBMClustering {
	public static Logger logger = LogManager.getLogger(RBMClustering.class);

	public int nVisible;
	public int nHidden;
	public double[][] W;
	public double[] hBias;
	public double[]	vBias;
	public Random rng;

	public RBMClustering(int nVisible, int nHidden, double[][] W, double[] hBias, double[] vBias, Random rng){
		this.nVisible = nVisible;
		this.nHidden = nHidden;

		if(rng == null){
			this.rng = new Random();
		}else{
			this.rng = rng;
		}

		if(W == null){
			this.W = new double[this.nHidden][this.nVisible];

			//a用来限制权值产生范围，这个范围有没有什么依据？
			double a = 1.0/this.nVisible;
			for(int i = 0;i < this.nHidden;i++){
				for(int j = 0;j < this.nVisible;j++){
					this.W[i][j] = uniform(a);
				}
			}
		}else{
			this.W = W;
		}

		//可见层隐含层偏置设为0。
		if(hBias == null){
			this.hBias = new double[this.nHidden];
			for(int i = 0;i < this.nHidden;i++){
				this.hBias[i] = 0;
			}
		}else{
			this.hBias = hBias;
		}

		if(vBias == null){
			this.vBias = new double[this.nVisible];
			for(int i = 0;i < this.nVisible;i++){
				this.vBias[i] = 0;
			}
		}else{
			this.vBias = vBias;
		}
	}
	/**
	 *
	 * @param input
	 * @param learningRate
	 * @param k
	 */
	public void cdk(Map<Integer, Integer> input, double learningRate, int k){
		double[] phMean = new double[nHidden];
		int[] phSample = new int[nHidden];

		double[] nvMean = new double[nVisible];
		Map<Integer, Integer> nvSample = new HashMap<>();

		double[] nhMean = new double[nHidden];
		int[] nhSample = new int[nHidden];

		sampleHGivenV(input, phMean, phSample);
		for(int i = 0;i < phSample.length;i++){
		}
		//执行k次吉布斯采样。
		for(int step = 0;step < k;step++){
			if(step == 0){
				gibbsHVH(phSample, nvMean, nvSample, nhMean, nhSample);
			}else{
				gibbsHVH(nhSample, nvMean, nvSample, nhMean, nhSample);
			}
		}

		//梯度下降更新权值及偏置
		for(int i = 0;i < nHidden;i++){
			for(int j: input.keySet()){
				W[i][j] += learningRate * (phMean[i] * input.get(j) - nhMean[i] * nvSample.get(j));
			}
			hBias[i] += learningRate * (phMean[i] - nhMean[i]);
		}
		for(int j: input.keySet()){
			vBias[j] += learningRate * (input.get(j) - nvSample.get(j));
		}
}

	public double propUp(Map<Integer, Integer> v, double[] w, double b){
		double preSigmoidActivation = 0.0;
		for(int j: v.keySet()){
			preSigmoidActivation += w[j] * v.get(j);
		}
		preSigmoidActivation += b;
		return sigmoid(preSigmoidActivation);
	}

	public double propDown(int[] h, int j, double b){
		double preSigmoidActivation = 0.0;
		for(int i = 0;i < h.length;i++){
			preSigmoidActivation += this.W[i][j] * h[i];
		}
		preSigmoidActivation += b;

		return sigmoid(preSigmoidActivation);
	}

	/**
	 * 两次吉布斯采样。P(v|h),P(h|v)
	 */
	public void gibbsHVH(int[] hSample, double[] nvMean, Map<Integer, Integer> nvSample, double[] nhMean, int[] nhSample){
		sampleVGivenH(hSample,nvMean,nvSample);
		sampleHGivenV(nvSample, nhMean,nhSample);
	}

	public void sampleVGivenH(int[] hSample, double[] mean, Map<Integer, Integer> sample){
		for(int i = 0;i < nVisible;i++){
			mean[i] = propDown(hSample, i, this.vBias[i]);
			//用随机数rng生成二项分布随机数。
			int oz = binomial(mean[i], rng);
			sample.put(i, oz);
		}
	}

	public void sampleHGivenV(Map<Integer, Integer> vSample, double[] mean, int[] sample){
		for(int i = 0;i < nHidden;i++){
			mean[i] = propUp(vSample, W[i], hBias[i]);
			//binomial函数应该就是在0-1分布上进行吉布斯采样。
			sample[i] = binomial(mean[i], rng);
		}
	}

	//以一定概率根据mean生成0/1随机数(0-1分布吉布斯采样)。
	private int binomial(double mean, Random rng){
		double d = 0.0;
		if((d = rng.nextDouble()) < mean){
			return 1;
		}else{
			return 0;
		}
	}

	private double sigmoid(double preSigmoidActivation){
		return 1.0/(1.0 + Math.exp(-preSigmoidActivation));
	}

	private double uniform(double a){
		return uniform(-a, a);
	}

	//在(x,y)区间内产生随机数。
	private double uniform(double x, double y) {
		return x + Math.random() * (y - x);
	}

	public static RBMTheta train(){
		Random rng = new Random();
		double learningRate = 0.1;
		int nVisible = 2;
		int nHidden = 10;
		RBMClustering rbm = new RBMClustering(nVisible, nHidden, null, null, null, rng);
		String inputPath = "/Users/liuxl/Root/DeepLearning/RBM/data/trainingdata.dat";

		Map<Integer, Map<Integer, Integer>> inputMap = new HashMap<>();
		BufferedReader reader = null;
		try{
			reader = new BufferedReader(new FileReader(inputPath));
			String lineString = null;
			int line = 0;
			while((lineString = reader.readLine()) != null){
				String[] inputStrings = lineString.split(" ");
				Map<Integer, Integer> input = new HashMap<>();
				input.put(0, Integer.parseInt(inputStrings[0]));
				input.put(1, Integer.parseInt(inputStrings[1]));
				inputMap.put(line++, input);
			}
			reader.close();
		}catch(IOException e){
			e.printStackTrace();
		}

		for(int i = 0;i < 100;i++){
			rbm.cdk(inputMap.get(i), learningRate, 10000);
		}

		RBMTheta rbmTheta = new RBMTheta();
		rbmTheta.W = rbm.W;
		rbmTheta.vBias = rbm.vBias;
		rbmTheta.hBias = rbm.hBias;
		return rbmTheta;
	}

	public void test(){

	}

	public static void main(String[] args){
		System.out.println("start...");
		Random rng = new Random();
		RBMTheta theta = train();
		Map<Integer, Integer> testMap1 = new HashMap<>();
		testMap1.put(0, 0);
		testMap1.put(1, 0);
		Map<Integer, Integer> testMap2 = new HashMap<>();
		testMap2.put(0, 100);
		testMap2.put(1, 100);
		Map<Integer, Integer> testMap3 = new HashMap<>();
		testMap3.put(0, 1);
		testMap3.put(1, 200);
		Map<Integer, Integer> testMap4 = new HashMap<>();
		testMap4.put(0, 45);
		testMap4.put(1, 50);
		double[] mean = new double[10];
		int[][] hSample = new int[4][10];

		RBMClustering rbm = new RBMClustering(2, 10, theta.W, theta.hBias, theta.vBias, rng);
		rbm.sampleHGivenV(testMap1, mean, hSample[0]);
		rbm.sampleHGivenV(testMap2, mean, hSample[1]);
		rbm.sampleHGivenV(testMap3, mean, hSample[2]);
		rbm.sampleHGivenV(testMap4, mean, hSample[3]);
		for(int i = 0;i < hSample.length;i++){
			int temp[] = hSample[i];
			System.out.print("第" + i + "条测试数据结果:(");
			for(int j = 0;j < temp.length;j++){
				System.out.print(temp[j]);
				if(j != temp.length - 1){
					System.out.print(",");
				}else{
					System.out.println(")");
				}
			}
		}
	}
}
