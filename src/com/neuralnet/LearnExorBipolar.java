package com.neuralnet;

public class LearnExorBipolar {

	public static void main(String[] args) {
		
		NeuralNet net = new NeuralNet(2, 4, 1, 0.2, 0.05, 0.9, true);

		double[][] exor_in_bipolar = new double[][] { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };

		double[][] exor_out_bipolar = new double[][] { { -1 }, { 1 }, { 1 }, { -1 } };

		net.train(exor_in_bipolar, exor_out_bipolar);

		//System.out.println("NeuralNet " + net);

		// Predict output

		net.think(new double[][] { { -1, -1 } });

		System.out.println("Input : -1,-1 Output : " + net.getOutput()[0][0] + " Expected : -1");

		net.think(new double[][] { { 1, -1 } });

		System.out.println("Input: 1,-1 , Output : " + net.getOutput()[0][0] + " Expected : 1");
	}

}
