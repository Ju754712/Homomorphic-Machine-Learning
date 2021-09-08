from layer import Layer
from scipy import signal
import numpy as np
import math 

class DropoutLayer(Layer):
    def __init__(self, rate):
        self.rate = rate



    # returns output for a given input
    def forward_propagation(self, input):
        self.mask = np.random.binomial(1,1-self.rate,input.shape) / (1-self.rate)
        return self.mask * input 

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        return self.mask * output_error 