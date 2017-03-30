package perceptron;

public class Layer {
	final float[] layer;

	public Layer(int size) {
		layer = new float[size];
	}
	
	public Layer(float[] layer){
        this.layer = layer;
    }
	
	public void set(int i, float f){
		layer[i] = f;
	}
	
	public int size(){
		return layer.length;
	}
	
	public void add(int i, float f){
		layer[i] += f;
	}
	
	public float get(int i){
		return layer[i];
	}
}
