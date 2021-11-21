from layer import Layer
import numpy as np
from mpmath import *
mp.dps=300000
from numba import njit

# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    
    def forward_propagation_more(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output
    def forward_propagation_ckks(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
    def backward_propagation_more(self, output_error, learning_rate):
        input_error = np.zeros(self.input.shape, dtype = object)
        act = self.activation_prime(self.input)
        for i in range(self.input.shape[0]):
            for j in range(self.input.shape[1]):
                
                input_error[i,j] = np.matmul(act[i,j] , output_error[i,j])
        return input_error
