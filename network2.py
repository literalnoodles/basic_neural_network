from typing import Tuple
import numpy as np
from copy import deepcopy

from helper import len_data, load_data, shuffle, printProg

# np.random.seed(0)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

class Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(1, size) for size in sizes[1:]]
        # each element consist of m row and n column (m -> number of node, n -> number of vector input)
        self.weights = [np.random.randn(size_input, size_layer)
                        for size_input, size_layer in zip(sizes[:-1],sizes[1:])] 
    
    def feed_forward(self, a):
        """
        It takes the input, multiplies it by the weights, adds the biases, and then applies the sigmoid
        function to the result
        
        :param a: input to the network
        :return: The output of the network.
        """
        # calculate the output of this network
        for weight, bias in zip(self.weights, self.biases):
            z = a @ weight + bias
            a = sigmoid(z)
        return a

    def back_prop(self, x: np.ndarray, y: np.ndarray):
        """
        We calculate the error in the output layer, then we calculate the error in the previous layer,
        and so on until we reach the input layer
        
        :param x: the input data
        :param y: the output of the network
        :return: The gradient of the cost function with respect to the weights and biases.
        """
        # nabla_w and nabla_b are for calculate gradient descent of all weights and biases
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # this is for storing all active layer
        actives = [x]
        # this is for storing all z (which is equal wx + b)
        zs = []
        # first calculate all active layer by feedforward
        for w, b in zip(self.weights, self.biases):
            z = actives[-1] @ w + b
            active = sigmoid(z)
            zs.append(z)
            actives.append(active)
        
        # calculate error in the output layer
        last_layer_delta = (self.calculate_cost_gradient(actives[-1], y) * sigmoid_derivative(zs[-1])).T
        # calculate nabla_w, nabla_b in all layer
        delta_z = last_layer_delta

        nabla_w[-1] = actives[-2].T @ delta_z.T
        nabla_b[-1] = delta_z.sum(1)
        for i in range(2, self.num_layers):
            z = zs[-i]
            delta_z = (self.weights[-i + 1] @ delta_z) * sigmoid_derivative(z).T
            delta_w = actives[-i-1].T @ delta_z.T
            delta_b = delta_z.sum(1)
            nabla_w[-i] = delta_w
            nabla_b[-i] = delta_b

        return nabla_w, nabla_b

    def calculate_cost_gradient(self, output_active_layer, y):
        """
            calculate the cost gradient of the last layer using quadratic function (not MSE)
        """
        return (output_active_layer - y)

    def measure(self, training_set):
        wrong_pred = 0
        for x, y in training_set:
            y_train = np.argmax(self.feed_forward(x))
            y_true = np.argmax(y)
            if y_train != y_true:
                wrong_pred += 1
        return wrong_pred / len(training_set)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        x, y = test_data
        test_results = np.argmax(self.feed_forward(x), axis=1)
        return np.count_nonzero(y == test_results)


    def stoc_grad(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        The function takes the training data, the number of epochs, the mini batch size, the learning
        rate, and the test data as input. It then shuffles the training data, creates mini batches from
        the shuffled training data, and updates the mini batches
        
        :param training_data: a list of tuples (x, y) representing the training inputs and the desired
        outputs
        :param epochs: number of times to go through the training data
        :param mini_batch_size: The size of the mini-batches to use when sampling
        :param eta: learning rate
        :param test_data: If provided, then the network will be evaluated against the test data after
        each epoch, and partial progress printed out. This is useful for tracking progress, but slows
        things down substantially
        """
        n = len_data(training_data)
        if test_data: n_test = len_data(test_data)

        # print(f'Cost: {self.measure(training_data)}')
        for j in range(epochs):
            # shuffle the training_data
            training_data = shuffle(training_data)
            # create mini_batches with size mini_batch_size from the shuffle training set
            mini_batches = [
                (training_data[0][i: i + mini_batch_size], training_data[1][i: i + mini_batch_size])
                for i in range(0, n, mini_batch_size)]
            
            for i, mini_batch in enumerate(mini_batches):
                self.update_mini_batch(mini_batch, eta)
                printProg(i, len(mini_batches), update_rate=50, endWith='')
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f'Epoch {j} completed - Cost: {self.measure(training_data)}')
    
    def update_mini_batch(self, mini_batch: Tuple[np.ndarray], eta: float):
        """
        > For each mini-batch, we calculate the gradient of the cost function with respect to the
        network's weights and biases, and update the weights and biases accordingly
        
        :param mini_batch: a list of tuples (x, y)
        :param eta: learning rate
        """
        # update weight and bias for network
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # loop in mini_batch
        x, y = mini_batch
        delta_nabla_w, delta_nabla_b = self.back_prop(x, y)
        nabla_w = [(w + delta_w) for w, delta_w in zip(nabla_w, delta_nabla_w)]
        nabla_b = [(b + delta_b) for b, delta_b in zip(nabla_b, delta_nabla_b)]
        self.weights = [w - eta * delta_w / len(mini_batch) for w, delta_w in zip(self.weights, nabla_w)]
        self.biases = [b - eta * delta_b / len(mini_batch) for b, delta_b in zip(self.biases, nabla_b)]


# nw = Network([784, 30, 10])
# training_data, validation_data, test_data = load_data()
# nw.stoc_grad(training_data, 30, 20, 1.0, test_data)

