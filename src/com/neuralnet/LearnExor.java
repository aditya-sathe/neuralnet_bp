package com.neuralnet;

public class LearnExor {

	public static void main(String[] args) {
		
		//NeuronLayer hidden = new NeuronLayer(4, 2);
		
		//NeuronLayer output = new NeuronLayer(1, 4);
		
		NeuralNet net = new NeuralNet(2, 4, 1, 0.2, 0.05, 0.9);
		
		double[][] exor_inset = new double[][]{
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1} };

	    double[][] exor_outs = new double[][]{
	            {0},
	            {1},
	            {1},
	            {0}
	    }; 
	    	    
	    net.train(exor_inset, exor_outs);
	    
	    System.out.println("NeuralNet " + net);
	    
//	    System.out.println("Output layer " + output);
	    
	    // Predict output
	    
	    net.think(new double[][]{{0, 0}});
	    
	    System.out.println("0,0 Output ->" + net.getOutput()[0][0]);
	    
	    net.think(new double[][]{{1, 0}});
	    
	    System.out.println("1,0 Output ->" + net.getOutput()[0][0]);
	}

}
