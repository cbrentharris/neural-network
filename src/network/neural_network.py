import numpy as np
import time

from network.math import mse, mse_prime, tanh, tanh_prime, sigmoid_prime, sigmoid


class Layer:
    def __init__(self, input_size, output_size, activation_function=tanh,
                 activation_prime=tanh_prime):
        """
        :param input_size: The input size to the layer
        :param output_size: The output size of the layer
        :param activation_function: The function used to activate the layer
        :param activation_prime: The derivative of the activation funct ion
        """
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.activation_function = activation_function
        self.activation_prime = activation_prime
        self.layer_input = None
        self.layer_output = None

    def forward_propagate(self, layer_input):
        """
        Calculate the output of this layer based on the input of the previous layer. Done by taking
        the dot product of the matrices.
        :param layer_input: The input from the previous layer
        :return: Activated output after weights and bias have been added
        """
        self.layer_input = layer_input
        self.layer_output = np.dot(self.layer_input, self.weights) + self.bias
        return self.activation_function(self.layer_output)

    def backward_propagate(self, output_error, learning_rate):
        activation_error = self.activation_prime(self.activation_function(self.layer_output)) * output_error
        input_error = np.dot(activation_error, self.weights.T)
        weights_error = np.dot(self.layer_input.T, activation_error)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


class NeuralNetwork:
    def __init__(self, input_layer_size, output_layer_size, hidden_layers, loss_function=mse, loss_prime=mse_prime):
        """
        :param input_layer_size: How large the initial input layer is in size. This should essentially be "num features"
        :param output_layer_size: The output layer size, essentially how many categories can be predicted.
        :param hidden_layers: This is an array of int, with each entry representing the input size to the hidden layer.
        :param loss_function: How to calculate loss of a prediction. Default is mean squared error
        :param loss_prime: Derivative of loss function.
        """
        layers = [
            Layer(input_layer_size, hidden_layers[0])
        ]

        if len(hidden_layers) < 1:
            raise Exception("Require at least 1 hidden layer")

        for i in range(0, len(hidden_layers) - 1):
            layers.append(Layer(hidden_layers[i], hidden_layers[i + 1]))

        layers.append(
            Layer(hidden_layers[-1], output_layer_size, activation_function=sigmoid, activation_prime=sigmoid_prime))
        self.loss_function = loss_function
        self.loss_prime = loss_prime
        self.layers = layers

    def fit(self, x_train, y_train, epochs=1000, learning_rate=0.01):
        """
        Take input examples to train against with their labeled output
        :param x_train: Input np arrays to train against
        :param y_train: Expected labeled output for each input to be able to minimize error
        :param epochs: How many iterations to train the network
        :param learning_rate: How quickly to adjust weights and biases in layers
        :return: None -- this method will update the network in place
        """
        samples = len(x_train)
        total_duration = 0
        for i in range(epochs):
            total_error = 0
            start = time.time()
            for j in range(samples):
                y_pred = x_train[j]
                for layer in self.layers:
                    y_pred = layer.forward_propagate(y_pred)

                total_error += self.loss_function(y_train[j], y_pred)

                error = self.loss_prime(y_train[j], y_pred)
                for layer in reversed(self.layers):
                    error = layer.backward_propagate(error, learning_rate)

            end = time.time()
            average_error = total_error / samples
            duration_in_seconds = end - start
            total_duration += duration_in_seconds
            epoch = i + 1
            average_duration = total_duration / epoch
            remaining_duration = average_duration * (epochs - epoch)
            print(
                f"epoch {epoch} / {epochs}, error={average_error:.3f}, duration for epoch in seconds={duration_in_seconds:.2f}, estimated remaining duration in minutes={remaining_duration / 60:.2f}")

    def predict(self, x_test):
        """
        Predict the labels of input features
        :param x_test: input features
        """
        y_pred = []

        for x in x_test:
            output = x
            for layer in self.layers:
                output = layer.forward_propagate(output)
            y_pred.append(output)

        return y_pred