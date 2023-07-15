import numpy as np
from random import random

# save the activations and derivatives
# implement backpropagation
# implement gradient descent
# implement combining train
# train our network with a dummy dataset
# make some predictions


class MLP:

    def __init__(self, num_inputs=3, num_hidden=None, num_outputs=2):
        if num_hidden is None:
            num_hidden = [3, 5]
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]
        print(layers)

        # initiate random weights

        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(w)

        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]), dtype=float)
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):

        activations = inputs
        self.activations[0] = inputs

        for i, w in enumerate(self.weights):
            # calculate net inputs
            net_inputs = np.dot(activations, w)

            # calculate the activations
            activations = self.sigmoid(net_inputs)
            self.activations[i + 1] = activations

        # a_3 = s(h_3)
        # h_3 = a_2 & W_2

        return activations

    def back_propagate(self, error, verbose=False):
        # dE/dW_i = (y - a_[i+1]) s' (h_[i+1])) a_i
        # s'(h_[i+1) = s(h_[i+1])(1 - s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1]
        # dE/dW_i = (y - a_[i+1]) s' (h_[i+1])) W_i s'(h_i) a_[i-1]

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]

            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T

            current_activations = self.activations[i]  # ndarray([0.1, 0.2]) --> ndarray([[0.1], [0.2]])
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)

            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            derivatives = self.derivatives[i]
            weights += derivatives*learning_rate

    def train(self, inputs, targets, epochs, learning_rate):

        for i in range(epochs):
            sum_error = 0
            for (input, target) in zip(inputs, targets):

                # perform forward prop
                output = self.forward_propagate(input)

                # calculate the error
                error = target - output

                # perform back prop
                self.back_propagate(error, verbose=False)

                # applying gradient descent
                self.gradient_descent(learning_rate)

                sum_error += self._mse(target, output)

            # report error for each epoch
            print("Error for epoch {}: {}".format(i, sum_error/len(inputs)))

    def _mse(self, target, output):
        return np.average((target - output)**2)


    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":

    # Dummy dataset
    inputs = np.array([[random() / 2 for _ in range(2)] for _ in range(1000)]) # array([[0.1, 0.2], [0.3, 0.4]])
    targets = np.array([[i[0] + i[1]] for i in inputs]) # Sum of features - target:[[0.3], [0.4]]

    # create an mlp
    mlp = MLP(2, [5], 1)

    # Train our mlp
    mlp.train(inputs, targets, 50, 0.1)

    # Dummy data
    input = np.array([0.2, 0.3])
    target = np.array([0.5])

    output = mlp.forward_propagate(input)
    print("\n\nOur network believes that {} + {} is equal to {}".format(input[0], input[1], output[0]))



