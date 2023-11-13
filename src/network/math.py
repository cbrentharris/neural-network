import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2));


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


def ReLU(x):
    return x * (x > 0)


def dReLU(x):
    return 1. * (x > 0)


def tanh(x):
    return np.tanh(x);


def tanh_prime(x):
    return 1 - np.tanh(x) ** 2;
