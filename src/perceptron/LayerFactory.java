package perceptron;

public class LayerFactory {
	public Layer create(int size){
		return new Layer(size);
	}
}
