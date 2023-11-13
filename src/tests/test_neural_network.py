import unittest
import random
import os
import numpy as np
from network.neural_network import NeuralNetwork
from network.math import mse


class NeuralNetworkTest(unittest.TestCase):

    def test_neural_network(self):
        current_file_directory = os.path.dirname(os.path.abspath(__file__))
        with open(current_file_directory + "/resources/Train.txt", "r") as data_file:
            training_and_test_data = data_file.readlines()

        x = []
        y = []
        for line in training_and_test_data:
            split = line.split("|")
            label = list(map(int, split[1].strip().replace("labels ", "").split()))
            features = list(map(int, split[2].strip().replace("features ", "").split()))
            x.append(features)
            y.append(label)

        combined = list(zip(x, y))
        random.shuffle(combined)

        split_index = int(len(combined) * (1 - .20))

        x_train, y_train = zip(*combined[:split_index])
        x_test, y_test = zip(*combined[split_index:])

        x_train = [np.array(x).reshape(1, len(x)).astype('float32') / 255 for x in x_train]
        x_test = [np.array(x).reshape(1, len(x)).astype('float32') / 255 for x in x_test]

        num_inputs = len(x[0])
        num_outputs = len(y[0])
        hidden_layers_neurons = [175, 87]
        learning_rate = 0.01
        network = NeuralNetwork(num_inputs, num_outputs, hidden_layers_neurons)
        network.fit(x_train, y_train, 35, learning_rate)
        error = 0
        for features, label in zip(x_test, y_test):
            predicted = network.predict([features])[0]
            expected = np.array(label).reshape(1, len(label))
            error += mse(predicted, expected)

        print(f"average error={error / len(x_test)}")


if __name__ == "__main__":
    unittest.main()
