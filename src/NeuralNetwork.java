import java.util.Arrays;

public class NeuralNetwork {
    public final double[][] hidden_layer_weights;
    public final double[][] output_layer_weights;
    private final int num_inputs;
    private final int num_hidden;
    private final int num_outputs;
    private final double learning_rate;
    private final double[][] bias;

    public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs, double[][] initial_hidden_layer_weights,
            double[][] initial_output_layer_weights, double learning_rate, double[][] bias) {
        // Initialise the network
        this.num_inputs = num_inputs;
        this.num_hidden = num_hidden;
        this.num_outputs = num_outputs;

        this.hidden_layer_weights = initial_hidden_layer_weights;
        this.output_layer_weights = initial_output_layer_weights;

        this.learning_rate = learning_rate;
        this.bias = bias;
    }

    // Calculate neuron activation for an input
    public double sigmoid(double input) {
        double output = 1 / (1 + Math.exp(-input));
        return output;
    }

    // Feed forward pass input to a network output
    public double[][] forward_pass(double[] inputs) {
        double[] hidden_layer_outputs = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
            double weighted_sum = bias[0][i];
            for (int j = 0; j < num_inputs; j++) {
                weighted_sum += inputs[j] * hidden_layer_weights[j][i];
            }
            double output = sigmoid(weighted_sum);
            hidden_layer_outputs[i] = output;
        }

        double[] output_layer_outputs = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            double weighted_sum = bias[0][i];
            for (int j = 0; j < num_hidden; j++) {
                weighted_sum += hidden_layer_outputs[j] * output_layer_weights[j][i]; 
            }
            double output = sigmoid(weighted_sum);
            output_layer_outputs[i] = output;
        }
        return new double[][] { hidden_layer_outputs, output_layer_outputs };
    }

    public double[][][] backward_propagate_error(double[] inputs, double[] hidden_layer_outputs,
            double[] output_layer_outputs, int desired_output) {

        // Calculate the error at the output layer
        double[] output_errors = new double[num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            double output = output_layer_outputs[i];
            double error = output * (1 - output) * (i == desired_output ? 1 - output : -output);
            output_errors[i] = error;
        }

        // Calculate the error at the hidden layer
        double[] hidden_errors = new double[num_hidden];
        for (int i = 0; i < num_hidden; i++) {
            double output = hidden_layer_outputs[i];
            double error = 0;
            for (int j = 0; j < num_outputs; j++) {
                error += output_layer_weights[i][j] * output_errors[j];
            }
            hidden_errors[i] = output * (1 - output) * error;
        }

        // Calculate the weight deltas for the output and hidden layers
        double[][] delta_output_layer_weights = new double[num_hidden][num_outputs];
        double[][] delta_hidden_layer_weights = new double[num_inputs][num_hidden];
        double[] delta_bias_hidden = new double[num_hidden];
        double[] delta_bias_output = new double[num_outputs];


        for (int i = 0; i < num_hidden; i++) {
            for (int j = 0; j < num_outputs; j++) {
                delta_output_layer_weights[i][j] = hidden_layer_outputs[i] * output_errors[j];
                delta_bias_output[j] = output_errors[j];
            }
        }
        for (int i = 0; i < num_inputs; i++) {
            for (int j = 0; j < num_hidden; j++) {
                delta_hidden_layer_weights[i][j] = inputs[i] * hidden_errors[j];
                delta_bias_hidden[j] += hidden_errors[j];
            }
        }

        return new double[][][] { delta_output_layer_weights, delta_hidden_layer_weights, new double[][]{delta_bias_output,delta_bias_hidden} };
    }

    public void update_weights(double[][] delta_output_layer_weights, double[][] delta_hidden_layer_weights,
            double[][] delta_bias) {
        for (int i = 0; i < num_hidden; i++) {
            for (int j = 0; j < num_outputs; j++) {
                output_layer_weights[i][j] += learning_rate * delta_output_layer_weights[i][j];
                bias[0][j] += learning_rate * delta_bias[0][j];
            }
        }

        for (int i = 0; i < num_inputs; i++) {
            for (int j = 0; j < num_hidden; j++) {
                hidden_layer_weights[i][j] += learning_rate * delta_hidden_layer_weights[i][j];
                bias[1][j] += learning_rate * delta_bias[1][j];
            }
        }

    }

    public void train(double[][] instances, int[] desired_outputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            System.out.println("epoch = " + epoch);
            int[] predictions = new int[instances.length];
            for (int i = 0; i < instances.length; i++) {
                double[] instance = instances[i];
                double[][] outputs = forward_pass(instance);
                double[][][] delta_weights = backward_propagate_error(instance, outputs[0], outputs[1],
                        desired_outputs[i]);
                int predictedClass = -1;
                double guess = 0;
                for (int j = 0; j < num_outputs; j++) {
                    if (outputs[1][j] > guess) {
                        predictedClass = j;
                        guess = outputs[1][j];
                    }
                }
                predictions[i] = predictedClass;

                // We use online learning, i.e. update the weights after every instance.
                update_weights(delta_weights[0], delta_weights[1], delta_weights[2]);
            }

            // Print new weights
            System.out.println("Hidden layer weights \n" + Arrays.deepToString(hidden_layer_weights));
            System.out.println("Output layer weights  \n" + Arrays.deepToString(output_layer_weights));

            // TODO: Print accuracy achieved over this epoch
            int correct = 0;
            for (int j = 0; j < instances.length; j++) {
                if (predictions[j] == desired_outputs[j]) {
                    correct++;
                }
            }
            double acc = (double) correct / (double) desired_outputs.length;
            System.out.println("acc = " + acc);
        }
    }

    public int[] predict(double[][] instances) {
        int[] predictions = new int[instances.length];
        for (int i = 0; i < instances.length; i++) {
            double[] instance = instances[i];
            double[][] outputs = forward_pass(instance);
            int predictedClass = -1;
            double guess = 0;
            for (int j = 0; j < num_outputs; j++) {
                if (outputs[1][j] > guess) {
                    predictedClass = j;
                    guess = outputs[1][j];
                }
            }
            predictions[i] = predictedClass;

        }
        return predictions;
    }

}
