package com.neuralnet;

import static com.neuralnet.MatrixUtil.apply;
import static com.neuralnet.NNMath.*;

import java.io.FileWriter;

import com.neuralnet.NeuronLayer.ActivationFunctionType;

/**
 * A two layer neuralnet.
 */
public class NeuralNet {

	private final NeuronLayer layer1, layer2;
	private double[][] outputLayer1;
	private double[][] outputLayer2;
	private final double learningRate;
	private double totalerr = 0;
	private double momentum = 0;

	public NeuralNet(int numOfInputs, int numOfHiddenLayerNeurons, int numOfOutputLayerNeurons, double learningRate,
			double error, double momentum, boolean useBipolarInputs) {

		if (useBipolarInputs = false) {
			this.layer1 = new NeuronLayer(numOfHiddenLayerNeurons, numOfInputs);
			this.layer2 = new NeuronLayer(numOfOutputLayerNeurons, numOfHiddenLayerNeurons);
		} else {
			this.layer1 = new NeuronLayer(ActivationFunctionType.TANH, numOfHiddenLayerNeurons, numOfInputs);
			this.layer2 = new NeuronLayer(ActivationFunctionType.TANH, numOfOutputLayerNeurons,
					numOfHiddenLayerNeurons);
		}
		this.learningRate = learningRate;
		this.totalerr = error;
		this.momentum = momentum;
	}

	/**
	 * Forward propagation
	 * 
	 * @param inputs
	 */
	public void applyPass(double[][] inputs) {
		outputLayer1 = apply(matrixMultiply(inputs, layer1.weights), layer1.activationFunction); // 4x5
		outputLayer1 = addBias(outputLayer1);
		outputLayer2 = apply(matrixMultiply(outputLayer1, layer2.weights), layer2.activationFunction); // 4x1
	}

	public void think(double[][] inputs) {
		inputs = addBias(inputs);
		applyPass(inputs);
	}

	/**
	 * Method to train the neuralnet
	 */
	public void train(double[][] inputs, double[][] outputs) {

		inputs = addBias(inputs);
		double[][] previousAdjustmentLayer1 = null;
		double[][] previousAdjustmentLayer2 = null;
		FileWriter filewriter = null;
		try {
			filewriter = new FileWriter("out.csv");

			for (int epochs = 1;; epochs++) {
				// pass the training set through the network
				applyPass(inputs); // 4x3

				// adjust weights by error * input * (derivative of output)

				// calculate the error for layer 2
				// (the difference between the desired output and predicted output for each of
				// the training inputs)
				double[][] errorLayer2 = matrixSubtract(outputs, outputLayer2);
				double[][] deltaLayer2 = scalarMultiply(errorLayer2,
						apply(outputLayer2, layer2.activationFunctionDerivative));

				// calculate the error for layer 1
				// (by looking at the weights in layer 1, we can determine by how much layer 1
				// contributed to the error in layer 2)

				double[][] errorLayer1 = matrixMultiply(deltaLayer2, matrixTranspose(layer2.weights));
				double[][] deltaLayer1 = scalarMultiply(errorLayer1,
						apply(outputLayer1, layer1.activationFunctionDerivative));

				// Calculate how much to adjust the weights by
				// Since we are dealing with matrices, we handle the division by multiplying
				// the delta output sum with the inputs' transpose!

				double[][] adjustmentLayer1 = matrixMultiply(matrixTranspose(inputs), deltaLayer1);
				double[][] adjustmentLayer2 = matrixMultiply(matrixTranspose(outputLayer1), deltaLayer2);

				// apply learning rate
				adjustmentLayer1 = MatrixUtil.apply(adjustmentLayer1, (x) -> learningRate * x);
				adjustmentLayer2 = MatrixUtil.apply(adjustmentLayer2, (x) -> learningRate * x);

				// remove extra column
				adjustmentLayer1 = removeLastColumn(adjustmentLayer1);

				// apply momentum
				if (!(epochs == 1)) {
					adjustmentLayer1 = NNMath.matrixAdd(adjustmentLayer1,
							MatrixUtil.apply(previousAdjustmentLayer1, (x) -> momentum * x));
					adjustmentLayer2 = NNMath.matrixAdd(adjustmentLayer2,
							MatrixUtil.apply(previousAdjustmentLayer2, (x) -> momentum * x));
				}
				// adjust the weights
				this.layer1.adjustWeights(adjustmentLayer1);
				this.layer2.adjustWeights(adjustmentLayer2);

				double err = calculateTotalError(errorLayer1, errorLayer2);
				/*if (epochs % 10 == 0) {
					System.out.println("---Training epochs of " + epochs + " Error " + err);
					filewriter.append(epochs + "," + err + "\n");
				}*/
				if (err < totalerr) {
					System.out.println("Total Training epochs of " + epochs + " Error " + err);
					filewriter.append(epochs + "," + err);
					break;
				}
				previousAdjustmentLayer1 = adjustmentLayer1;
				previousAdjustmentLayer2 = adjustmentLayer2;
			}
		} catch (Exception e) {
			e.printStackTrace();
		} finally {
			try {
				filewriter.flush();
				filewriter.close();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * Calculate mean squared error
	 * 
	 * @param errorLayer1
	 * @param errorLayer2
	 * @return
	 */
	private double calculateTotalError(double[][] errorLayer1, double[][] errorLayer2) {
		double error1 = 0, error2 = 0;

		for (int i = 0; i < errorLayer1.length; i++) {
			for (int j = 0; j < errorLayer1[0].length; j++) {
				error1 = error1 + Math.pow(errorLayer1[i][j], 2);
			}
		}
		for (int i = 0; i < errorLayer2.length; i++) {
			for (int j = 0; j < errorLayer2[0].length; j++) {
				error2 = error2 + Math.pow(errorLayer2[i][j], 2);
			}
		}
		return (error1 + error2) / 2;
	}

	private double[][] removeLastColumn(double[][] inputs) {
		int row = inputs.length;
		int col = inputs[0].length;
		int colRemove = inputs[0].length - 1;

		double[][] newArray = new double[row][col - 1]; // new Array will have one column less
		for (int i = 0; i < row; i++) {
			for (int j = 0, currColumn = 0; j < col; j++) {
				if (j != colRemove) {
					newArray[i][currColumn++] = inputs[i][j];
				}
			}
		}
		return newArray;
	}

	private double[][] addBias(double[][] inputs) {
		double[][] result = new double[inputs.length][inputs[0].length + 1];
		for (int i = 0; i < result.length; i++) {
			for (int j = 0; j < result[0].length; j++) {
				result[i][j] = (j == result[0].length - 1) ? 1 : inputs[i][j];
			}
		}
		return result;
	}

	public double[][] getOutput() {
		return outputLayer2;
	}

	@Override
	public String toString() {
		String result = "Layer 1\n";
		result += layer1.toString();
		result += "Layer 2\n";
		result += layer2.toString();

		if (outputLayer1 != null) {
			result += "Layer 1 output\n";
			result += MatrixUtil.matrixToString(outputLayer1);
		}

		if (outputLayer2 != null) {
			result += "Layer 2 output\n";
			result += MatrixUtil.matrixToString(outputLayer2);
		}

		return result;
	}
}
