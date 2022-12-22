import java.util.Arrays;
import java.util.Random;

public class NeuralNetwork {
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[][] inputToHiddenWeights;
    private double[][] hiddenToOutputWeights;
    private double[] hiddenBiases;
    private double[] outputBiases;
    private Random random;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        inputToHiddenWeights = new double[inputSize][hiddenSize];
        hiddenToOutputWeights = new double[hiddenSize][outputSize];
        hiddenBiases = new double[hiddenSize];
        outputBiases = new double[outputSize];
        random = new Random();
        initWeights();
    }

    public double[] forward(double[] input) {
        double[] hidden = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++) {
            hidden[i] = hiddenBiases[i];
            for (int j = 0; j < inputSize; j++) {
                hidden[i] += input[j] * inputToHiddenWeights[j][i];
            }
            hidden[i] = sigmoid(hidden[i]);
        }
        double[] output = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            output[i] = outputBiases[i];
            for (int j = 0; j < hiddenSize; j++) {
                output[i] += hidden[j] * hiddenToOutputWeights[j][i];
            }
            output[i] = sigmoid(output[i]);
        }
        return output;
    }

    public void train(double[][] inputs, double[][] targets, int epochs, double learningRate) {
        for (int i = 0; i < epochs; i++) {
            for (int j = 0; j < inputs.length; j++) {
                double[] output = forward(inputs[j]);
                double[] error = new double[outputSize];
                for (int k = 0; k < outputSize; k++) {
                    error[k] = targets[j][k] - output[k];
                }
                double[] hiddenError = new double[hiddenSize];
                for (int k = 0; k < hiddenSize; k++) {
                    hiddenError[k] = 0;
                    for (int l = 0; l < outputSize; l++) {
                        hiddenError[k] += error[l] * hiddenToOutputWeights[k][l];
                    }
                }
                for (int k = 0; k < outputSize; k++) {
                    outputBiases[k] += learningRate * error[k];
                    for (int l = 0; l < hiddenSize; l++) {
                        hiddenToOutputWeights[l][k] += learningRate * error[k] * hidden[l];
                    }
                }
                for (int k = 0; k < hiddenSize; k++) {
                    hiddenBiases[k] += learningRate * hiddenError[k];
                    for (int l = 0; l < inputSize; l++) {
                        inputToHiddenWeights[l][k] += learningRate * hiddenError[k] * inputs[j][l];
                    }
                }
            }
        }
    }

    private void initWeights() {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                inputToHiddenWeights[i][j] = random.nextGaussian();
            }
        }
        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                hiddenToOutputWeights[i][j] = random.nextGaussian();
            }
        }
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}
