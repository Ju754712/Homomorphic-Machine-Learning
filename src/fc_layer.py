from layer import Layer
import numpy as np
from schemes.more import MoreScheme

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
        # self.output = dot_wb(self.input, self.weights, self.bias)
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        # input_error = dot_wob(output_error, self.weights.T)
        # weights_error = dot_wob(self.input.T, output_error)
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

class BFV_FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) -0.5

    def forward_propagation(self, input_vector):
        # Decomposition of dot product, where addition and multiplication are exchanged with plain arithmetics for bfv
        i = 0
        enc_channels = []
        while i < output_size:
            y = input_vector.dot(weights[:,i])
            enc_channels.append(y)
        self.output = ts.BFVVector.pack_vectors(enc_channels)


    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

class MORE_FCLayer(Layer):
    def __init__(self, input_size, output_size, more):
        self.weights = np.zeros((input_size, output_size), dtype = object)  
        self.bias = np.zeros((1, output_size), dtype = object)   
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.zeros((1,self.weights.shape[1]),dtype=object)
        i = 0
        while i < self.weights.shape[1]:
            j = 0 
            while j < self.input.shape[1]:
                self.output[0,i] += np.matmul(self.input[0,j], self.weights[j,i])
                j+=1
            self.output[0,i] += self.bias[0,i]
            i+=1
        return self.output



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