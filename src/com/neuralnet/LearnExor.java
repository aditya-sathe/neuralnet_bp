package com.neuralnet;

public class LearnExor {

	public static void main(String[] args) {

		NeuralNet net = new NeuralNet(2, 4, 1, 0.2, 0.05, 0.0, false);

		double[][] exor_inset = new double[][] { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };

		double[][] exor_outs = new double[][] { { 0 }, { 1 }, { 1 }, { 0 } };

		net.train(exor_inset, exor_outs);

		//System.out.println("NeuralNet " + net);

		// Predict output

		net.think(new double[][] { { 0, 0 } });

		System.out.println("0,0 Output : " + net.getOutput()[0][0] + " Expected: 0");

		net.think(new double[][] { { 1, 0 } });

		System.out.println("1,0 Output : " + net.getOutput()[0][0] + " Expected: 1");
	}

}
