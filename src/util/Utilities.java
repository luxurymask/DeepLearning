package util;

public class Utilities {
	public static float sigmoid(float x){
		return (float)(1.0f / (1.0f + Math.exp(-x)));
	}
	
	public static double sigmoid(double x){
		return 1.0 / (1.0 + Math.exp(-x));
	}
	
	public static int step(float x){
		return x >= 0 ? 1 : 0;
	}
}
