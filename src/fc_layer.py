from layer import Layer
from schemes.bfv import mul_plain, add_plain
import numpy as np

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = dot_wb(self.input, self.weights, self.bias)
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = dot_wob(output_error, self.weights.T)
        weights_error = dot_wob(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

class BFV_FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) -0.5

    def forward_propagation(self, input_data):
        # Decomposition of dot product, where addition and multiplication are exchanged with plain arithmetics for bfv
        output

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error


def dot_wb(input, weights, bias):
    output = np.zeros((1,weights.shape[1]))
    i = 0
    while i < weights.shape[1]:
        j = 0 
        while j < input.shape[1]:
            output[0,i] += input[0,j] * weights[j,i]
            j+=1
        output[0,i] += bias[0,i]
        i+=1
    return output

def dot_wob(input, weights):
    output = np.zeros((1,weights.shape[1]))
    i = 0
    while i < weights.shape[1]:
        j = 0 
        while j < input.shape[1]:
            output[0,i] += input[0,j] * weights[j,i]
            j+=1
        i+=1
    return output