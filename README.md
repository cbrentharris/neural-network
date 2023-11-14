# neural-network

This repository hosts a custom implementation of a neural network in python
for pedagogical reasons.

It leverages numpy for matrix multiplication, but implements layers, hidden layers,
back prop, forward prop, activation functions and loss functions.

# tests

For testing purposes, it executes against MNIST handwritten digit input. Due to the size of the
input file, it is not checked in, but can be downloaded from kaggle.

The test based on the layer configuration and input filtering can take in the minutes to finish, simply
reduce the size of the hidden layers or size of train and test data to speed up the tests.