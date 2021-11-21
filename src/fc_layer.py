from layer import Layer
import numpy as np
from schemes.more import MoreScheme
import tenseal as ts
from numba import njit

# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.scale = None

    def encrypt_params_ckks(self, context):
        self.weights_ckks = np.zeros(self.weights.shape, dtype = object)  
        self.bias_ckks = np.zeros(self.bias.shape, dtype = object)   
        for i in range(self.weights_ckks.shape[1]):
            for j in range(self.weights_ckks.shape[0]):
                self.weights_ckks[j,i] = ts.ckks_vector(context,[self.weights[j,i]])
            self.bias_ckks[0,i] = ts.ckks_vector(context, [self.bias[0,i]]) 
    def encrypt_params_more(self, more):
        self.weights_more = np.zeros((self.weights.shape[0], self.weights.shape[1],2,2))  
        self.bias_more = np.zeros((self.bias.shape[0],self.bias.shape[1],2,2))   
        for i in range(self.weights_more.shape[1]):
            for j in range(self.weights_more.shape[0]):
                self.weights_more[j,i] = more.encrypt(self.weights[j,i])
            self.bias_more[0,i] = more.encrypt(self.bias[0,i]) 
    def decrypt_params_more(self,more): 
        for i in range(self.weights_more.shape[1]):
            for j in range(self.weights_more.shape[0]):
                self.weights[j,i] = more.decrypt(self.weights_more[j,i])
            self.bias[0,i] = more.decrypt(self.bias_more[0,i]) 

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        output = np.dot(self.input, self.weights) + self.bias
        return output

    def forward_propagation_ckks(self, input_vector):
        self.input = input_vector
        output = input_vector.matmul(self.weights) + self.bias[0]
        return output
    def forward_propagation_more(self, input_data):
        self.input = input_data
        output = np.zeros((1, self.weights.shape[1]), dtype=object)
        i = 0
        while i < self.weights.shape[1]:
            j = 0 
            while j < self.input.shape[1]:
                output[0,i] += self.input[0,j] * self.weights[j,i]
                j+=1
            output[0,i] += self.bias[0,i] * np.identity(2)
            i+=1
        return output

    def forward_propagation_ckks_encrypted(self, input_data):   

        self.input = input_data
        output = np.zeros((1,self.weights_ckks.shape[1]),dtype=object)
        i = 0
        while i < self.weights_ckks.shape[1]:
            j = 0 
            while j < self.input.shape[1]:
                output[0,i] += self.input[0,j].dot(self.weights_ckks[j,i])
                j+=1
            output[0,i] += self.bias_ckks[0,i]
            i+=1
        return output
    def forward_propagation_more_encrypted(self, input_data):
        self.input = input_data
        output = np.zeros((1,self.weights_more.shape[1],2,2))
        i = 0
        while i < self.weights_more.shape[1]:
            j = 0 
            while j < self.input.shape[1]:
                output[0,i] += matmul(self.input[0,j], self.weights_more[j,i])
                j+=1
            output[0,i] += self.bias_more[0,i]
            i+=1
        return output
    
    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    def backward_propagation_ckks(self, output_error, learning_rate):
        input_error = np.zeros(self.input.shape, dtype = object)
        weights_error = np.zeros(self.weights_ckks.shape, dtype = object)
        i = 0
        while i < self.weights_ckks.T.shape[1]:
            j = 0
            while j < output_error.shape[1]:
                input_error[0,i] += output_error[0,j].dot(self.weights_ckks.T[j,i])
                j += 1
            i += 1
        
        i = 0
    
        while i < weights_error.shape[0]:
            j = 0
            while j < weights_error.shape[1]:
                weights_error[i,j] += self.input.T[i,0].dot(output_error[0,j])
                j += 1
            i += 1

        # self.weights_ckks -= learning_rate * self.weights_error
        # self.bias_ckks -= learning_rate * output_error
        return input_error, weights_error
    
    def backward_propagation_more(self, output_error, learning_rate):
        print(self.input[0,0])
        print(output_error[0,0])
        print(learning_rate)
        input_error, weights_more, bias_more = backward_more(self.input, self.weights_more, self.bias_more, output_error, learning_rate)
    
        self.weights_more = weights_more
        self.bias_more = bias_more
        return input_error




@njit
def matmul(x,y):
    r = np.zeros((x.shape[0], y.shape[1]))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            for k in range(x.shape[1]):
                r[i,j] += x[i,k]*y[k,j]
    return r
# @njit
def backward_more(input, weights_more, bias_more,output_error, learning_rate):
    input_error = np.zeros(input.shape)
    weights_error = np.zeros(weights_more.shape)
    input_transpose = np.transpose(input, (1,0,2,3))
    weights_more_trans = np.transpose(weights_more,(1,0,2,3))
    print(input_error.shape, weights_error.shape, input_transpose.shape, weights_more_trans.shape)
    i = 0
    while i < weights_more_trans.shape[1]:
        j = 0
        while j < output_error.shape[1]:
            input_error[0,i] += matmul(output_error[0,j], weights_more_trans[j,i])
            j += 1
        i += 1
        
    j = 0
    while j < weights_error.shape[1]:
        i = 0
        while i < weights_error.shape[0]:
            weights_error[i,j] += matmul(input_transpose[i,0], output_error[0,j])
            weights_more[i,j] -= learning_rate *weights_error[i,j]
            i += 1
        bias_more[0,j] -=learning_rate*output_error[0,j]
        
        j += 1

    return input_error, weights_more, bias_more
        



