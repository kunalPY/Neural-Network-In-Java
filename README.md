# Neural Network in Java

This is a simple neural network model implemented in Java. It includes a single hidden layer and can be trained using backpropagation.

## Installation

To use this neural network model, you will need to have the following installed:

- Java 8 or higher
- A Java development environment (e.g. Eclipse, IntelliJ)

## Usage

The `NeuralNetwork` class is the main class of the model. It has the following methods:

- `NeuralNetwork(int inputSize, int hiddenSize, int outputSize)`: Constructs a new neural network with the specified number of input, hidden, and output units.
- `double[] forward(double[] input)`: Takes an input array and returns the output of the neural network.
- `void train(double[][] inputs, double[][] targets, int epochs, double learningRate)`: Trains the neural network using the specified input-target pairs, number of epochs, and learning rate.

Here is an example of how to use the `NeuralNetwork` class:
```java
NeuralNetwork model = new NeuralNetwork(2, 3, 1);
double[][] inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
double[][] targets = {{0}, {1}, {1}, {0}};
model.train(inputs, targets, 1000, 0.1);
double[] output = model.forward({1, 1});
System.out.println(Arrays.toString(output));
```
This example creates a neural network with 2 input units, 3 hidden units, and 1 output unit. It trains the network on the XOR function using 1000 epochs and a learning rate of 0.1. Finally, it prints the output of the network when given the input `{1, 1}`.

