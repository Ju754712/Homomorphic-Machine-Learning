from layer import Layer
import numpy as np
from schemes.more import MoreScheme
import tenseal as ts

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def encrypt_params_bfv(self, context):
        return
    def encrypt_params_ckks(self, context):
        self.weights_ckks = []
        print(self.weights.shape)
        for i in range(self.weights.shape[1]):
            self.weights_ckks.append(ts.ckks_vector(context,self.weights[:,i]))
        self.bias_ckks = ts.ckks_vector(context,self.bias[0])
    def encrypt_params_more(self, more):
        self.weights_more = np.zeros(self.weights.shape, dtype = object)  
        self.bias_more = np.zeros(self.bias.shape, dtype = object)   
        for i in range(self.weights_more.shape[1]):
            for j in range(self.weights_more.shape[0]):
                self.weights_more[j,i] = more.encrypt(self.weights[j,i])
            self.bias_more[0,i] = more.encrypt(self.bias[0,i]) 

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        # self.output = dot_wb(self.input, self.weights, self.bias)
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    def forward_propagation_ckks(self, input_vector):
        # Decomposition of dot product, where addition and multiplication are exchanged with plain arithmetics for bfv
        self.output = input_vector.matmul(self.weights) + self.bias[0]
        return self.output
    def forward_propagation_more(self, input_data):
        self.input = input_data
        self.output = np.zeros((1, self.weights.shape[1]), dtype=object)
        i = 0
        while i < self.weights.shape[1]:
            j = 0 
            while j < self.input.shape[1]:
                self.output[0,i] += self.input[0,j] * self.weights_more[j,i]
                j+=1
            self.output[0,i] += self.bias[0,i]
            i+=1
        return self.output

    def forward_propagation_more_encrypted(self, input_data):
        self.input = input_data
        self.output = np.zeros((1,self.weights_more.shape[1]),dtype=object)
        i = 0
        while i < self.weights_more.shape[1]:
            j = 0 
            while j < self.input.shape[1]:
                self.output[0,i] += np.matmul(self.input[0,j], self.weights_more[j,i])
                j+=1
            self.output[0,i] += self.bias[0,i]
            i+=1
        return self.output

    def forward_propagation_ckks_encrypted(self, input_vector):
        self.input_vector = input_vector
        self.output = input_vector.dot(self.weights) + self.bias
    
    
    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error, weights_error

    def backward_propagation_ckks(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

    def backward_propagation_ckks_encrypted(self, output_error, learning_rate):
        input_error = output_error * self.weights
        weights_error = self.input_vector* output_error
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error, weights_error

    def backward_propagation_more(self, output_error, learning_rate):
        self.input_error = np.zeros(self.input.shape, dtype = object)
        self.weights_error = np.zeros(self.weights.shape, dtype = object)
        i = 0
        while i < self.weights.T.shape[1]:
            j = 0
            while j < output_error.shape[1]:
                self.input_error[0,i] += np.matmul(output_error[0,j], self.weights.T[j,i])
                j += 1
            i += 1
        
        i = 0
    
        while i < self.input.shape[0]:
            j = 0
            while j < output_error.shape[0]:
                self.weights_error[i,j] += np.matmul(self.input.T[i,0], output_error[0,j])
                j += 1
            i += 1

        self.weights -= learning_rate * self.weights_error
        self.bias -= learning_rate * output_error
        return self.input_error, self.weights_error


        



