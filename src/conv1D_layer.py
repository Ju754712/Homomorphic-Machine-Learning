from layer import Layer
from scipy import signal
import numpy as np
from numba import njit, jit
from numba.types import pyobject
from math import floor, ceil 
import time
import tenseal as ts

## Math behind this layer can found at : 
## https://medium.com/@2017csm1006/forward-and-backpropagation-in-convolutional-neural-network-4dfa96d7b37e

# inherit from base class Layer
# This convolutional layer is always with stride 1
class Conv1DLayer(Layer):
    def __init__(self, input_shape, kernel, layer_depth, strides, padding = 'same'):
        self.input_shape = input_shape
        self.kernel = kernel
        self.layer_depth = layer_depth
        self.strides = strides
        self.scale = None
        
        if padding == 'same':
            self.padding = floor(kernel/2)
        elif padding == 'valid':
            self.padding = 0 #incorrect
        elif padding == 'full':
            self.padding = 0 #incorrect
        else:
            self.padding = padding
        
        #Initialize learnable parameters
        self.weights = np.random.rand(kernel, self.input_shape[1], layer_depth) - 0.5
        self.bias = np.random.rand(layer_depth) - 0.5
    def scale_params_bfv(self, scale):
        self.scale = scale
        self.weights_scaled = (self.weights*self.scale).astype(int)
        self.bias_scaled = (self.bias*self.scale).astype(int)

    def encrypt_params_bfv(self, context):
        if self.scale != None:
            self.weights_bfv = np.zeros(self.weights.shape, dtype=object)
            self.bias_bfv = np.zeros(self.bias.shape, dtype=object)
            for k in range(self.weights_bfv.shape[2]):
                for i in range(self.weights_bfv.shape[0]):
                    for j in range(self.weights_bfv.shape[1]):
                        self.weights_bfv[j,i,k] = ts.bfv_vector(context,[self.weights_scaled[j,i,k]])
            self.bias_bfv[k] = ts.bfv_vector(context, [self.bias_scaled[k]]) 
        else:
            print("Please scale up parameters first")

    def encrypt_params_ckks(self, context):
        self.weights_ckks = np.zeros(self.weights.shape, dtype = object)  
        self.bias_ckks = np.zeros(self.bias.shape, dtype = object)   
        for k in range(self.weights_ckks.shape[2]):
            for i in range(self.weights_ckks.shape[1]):
                for j in range(self.weights_ckks.shape[0]):
                    self.weights_ckks[j,i,k] = ts.ckks_vector(context,[self.weights[j,i,k]])
            self.bias_ckks[k] = ts.ckks_vector(context, [self.bias[k]]) 

    def encrypt_params_more(self, more):
        self.weights_more = np.zeros((self.weights.shape[0],self.weights.shape[1], self.weights.shape[2],2,2))  
        self.bias_more = np.zeros((self.bias.shape[0],2,2))   
        for k in range(self.weights_more.shape[2]):
            for i in range(self.weights_more.shape[1]):
                for j in range(self.weights_more.shape[0]):
                    self.weights_more[j,i,k] = more.encrypt(self.weights[j,i,k])
            self.bias_more[k] = more.encrypt(self.bias[k]) 

    # returns output for a given input
    def forward_propagation(self, input):
        self.input = input
        self.output = convolution(input = self.input, weights = self.weights, bias = self.bias, layer_depth = self.layer_depth, kernel = self.kernel, strides = self.strides, dilation = 1, z_padding = 0, padding = self.padding, a = 0) 
        return self.output
    
    def forward_propagation_bfv(self, input):
        self.input = input
        self.output = convolution(input = self.input, weights = self.weights_bfv, bias = self.bias_bfv, layer_depth = self.layer_depth, kernel = self.kernel, strides = self.strides, dilation = 1, z_padding = 0, padding = self.padding, a = 0) 
        return self.output

    def forward_propagation_ckks(self, input):
        self.input = input
        self.output = convolution_ckks(input = self.input, weights = self.weights, bias = self.bias, layer_depth = self.layer_depth, kernel = self.kernel, strides = self.strides, dilation = 1, z_padding = 0, padding = self.padding, a = 0) 
        return self.output

    def forward_propagation_more(self, input):
        self.input = input
        self.output = convolution_more(input = self.input, weights = self.weights, bias = self.bias, layer_depth = self.layer_depth, kernel = self.kernel, strides = self.strides, dilation = 1, z_padding = 0, padding = self.padding, a = 0) 
        return self.output
    

    

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate): 
        # Compute Filter and Bias derivative and Input Error     
        dWeights = compute_dWeights(input = self.input, weights = self.weights, output_error = output_error,  kernel = self.kernel, layer_depth = self.layer_depth, strides = self.strides, dilation = 1, z_padding = 0, padding = self.padding)
        in_error = compute_in_error(input_shape = self.input_shape, weights = self.weights, output_error = output_error, kernel = self.kernel, layer_depth = self.layer_depth, strides = self.strides, dilation = 1, padding = self.padding, z_padding = 0)
        dBias = compute_dBias(layer_depth = self.layer_depth, output_error = output_error)

        # Update params
        self.weights -= learning_rate*dWeights
        self.bias -= learning_rate*dBias

        # Return Input Error
        return in_error

class Conv1DTransposedLayer(Layer):
    def __init__(self, input_shape, kernel, layer_depth, strides, a, padding = 'same'):
        self.input_shape = input_shape
        self.a = a
        self.kernel = kernel
        self.layer_depth = layer_depth
        self.scale = None
        if(padding == 'same'):
            self.padding = floor(kernel/2)
        elif(padding == 'valid'):
            self.padding = 0
        else: 
            self.padding = padding
        if strides == 1:
            self.p = kernel-self.padding-1
        else:
            self.p = kernel-self.padding
        self.z_padding = strides - 1
        self.weights = np.random.rand(kernel, self.input_shape[1], layer_depth) - 0.5
        self.bias = np.random.rand(layer_depth) - 0.5

    def scale_params_bfv(self, scale):
        self.scale = scale
        self.weights_scaled = (self.weights*self.scale).astype(int)
        self.bias_scaled = (self.bias*self.scale).astype(int)

    def encrypt_params_bfv(self, context):
        if self.scale != None:
            self.weights_bfv = np.zeros(self.weights.shape, dtype=object)
            self.bias_bfv = np.zeros(self.bias.shape, dtype=object)
            for k in range(self.weights_bfv.shape[2]):
                for i in range(self.weights_bfv.shape[0]):
                    for j in range(self.weights_bfv.shape[1]):
                        self.weights_bfv[j,i,k] = ts.bfv_vector(context,[self.weights_scaled[j,i,k]])
            self.bias_bfv[k] = ts.bfv_vector(context, [self.bias_scaled[k]]) 
        else:
            print("Please scale up parameters first")

    def encrypt_params_ckks(self, context):
        self.weights_ckks = np.zeros(self.weights.shape, dtype = object)  
        self.bias_ckks = np.zeros(self.bias.shape, dtype = object)   
        for k in range(self.weights_ckks.shape[2]):
            for i in range(self.weights_ckks.shape[1]):
                for j in range(self.weights_ckks.shape[0]):
                    self.weights_ckks[j,i,k] = ts.ckks_vector(context,[self.weights[j,i,k]])
            self.bias_ckks[k] = ts.ckks_vector(context, [self.bias[k]]) 

    def encrypt_params_more(self, more):
        self.weights_more = np.zeros((self.weights.shape[0],self.weights.shape[1], self.weights.shape[2],2,2))  
        self.bias_more = np.zeros((self.bias.shape[0],2,2))   
        for k in range(self.weights_more.shape[2]):
            for i in range(self.weights_more.shape[1]):
                for j in range(self.weights_more.shape[0]):
                    self.weights_more[j,i,k] = more.encrypt(self.weights[j,i,k])
            self.bias_more[k] = more.encrypt(self.bias[k]) 

    # returns output for a given input
    def forward_propagation(self, input):
        self.input = input
        self.output= trans_convolution(input = self.input, weights = self.weights, bias = self.bias,  kernel = self.kernel, layer_depth = self.layer_depth, strides = 1, dilation = 1, z_padding = self.z_padding, padding = self.p, a = self.a)
        return self.output
    def forward_propagation_bfv(self, input):
        self.input = input
        self.output= trans_convolution(input = self.input, weights = self.weights_bfv, bias = self.bias_bfv,  kernel = self.kernel, layer_depth = self.layer_depth, strides = 1, dilation = 1, z_padding = self.z_padding, padding = self.p, a = self.a)
        return self.output
    def forward_propagation_ckks(self, input):
        self.input = input
        self.output= trans_convolution_ckks(input = self.input, weights = self.weights, bias = self.bias,  kernel = self.kernel, layer_depth = self.layer_depth, strides = 1, dilation = 1, z_padding = self.z_padding, padding = self.p, a = self.a)
        return self.output
    def forward_propagation_more(self, input):
        self.input = input
        self.output= trans_convolution_more(input = self.input, weights = self.weights, bias = self.bias,  kernel = self.kernel, layer_depth = self.layer_depth, strides = 1, dilation = 1, z_padding = self.z_padding, padding = self.p, a = self.a)
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        dWeights = compute_dWeights(input = self.input, weights = self.weights, output_error = output_error,  kernel = self.kernel, layer_depth = self.layer_depth, strides = 1, dilation = 1, z_padding = self.z_padding, padding = self.p)
        dBias = compute_dBias(layer_depth = self.layer_depth, output_error = output_error)
        in_error = compute_in_error(input_shape = self.input.shape, weights = self.weights, output_error = output_error, kernel = self.kernel, layer_depth = self.layer_depth, strides = 1, dilation = 1, padding = self.p, z_padding = self.z_padding)
        self.weights -= learning_rate*dWeights
        self.bias -= learning_rate*dBias
        return in_error


@njit
def convolution(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a):
    input_length = input.shape[0]
    output = np.zeros((floor((input_length+2*padding+(input_length-1)*z_padding+a-(kernel+(kernel-1)*(dilation-1)))/strides)+1,layer_depth))
    i = 0
    while i < output.shape[0]:  
        # print("Computing output ", i, ' by adding up: ')                 
        offset = i*strides-padding+1  
        j = 0
        while j < kernel:
            if((offset+j*dilation)/(z_padding+1) in range(input.shape[0])):
                # print('weight ', j, ' times Input ', int((offset+j*dilation)/(z_padding+1)))
                k = 0
                while k < layer_depth:
                    d = 0
                    while d < input.shape[1]:
                        
                        output[i,k] += weights[j,d,k] * input[int((offset+j*dilation)/(z_padding+1)),d]
                        d += 1
                    k += 1                    
            j += 1
        k = 0
        while k < layer_depth:
            output[i,k] += bias[k]
            k += 1
        i += 1

    return output

# Backup for Transposed

def convolution_ckks(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a):
    input_length = input.shape[0]
    output = np.zeros((floor((input_length+2*padding+(input_length-1)*z_padding+a-(kernel+(kernel-1)*(dilation-1)))/strides)+1,layer_depth), dtype=object)
    i = 0
    while i < output.shape[0]:                   
        offset = i*strides-padding+1  
        j = 0
        while j < kernel:
            if((offset+(j)*dilation)/(z_padding+1) in range(input.shape[0])):
                k = 0
                while k < layer_depth:
                    d = 0
                    while d < input.shape[1]:
                        output[i,k] += weights[j,d,k] * input[int((offset+j*dilation)/(z_padding+1)),d]
                        d += 1
                    k += 1                    
            j += 1
        k = 0
        while k < layer_depth:
            output[i,k] += bias[k]
            k += 1
        i += 1

    return output

@njit
def convolution_more(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a):
    input_length = input.shape[0]
    output = np.zeros((floor((input_length+2*padding+(input_length-1)*z_padding+a-(kernel+(kernel-1)*(dilation-1)))/strides)+1,layer_depth,2,2))
    idn = np.identity(2)
    i = 0
    while i < output.shape[0]:  
        # print("Computing output ", i, ' by adding up: ')                 
        offset = i*strides-padding+1  
        j = 0
        while j < kernel:
            if((offset+j*dilation)/(z_padding+1) in range(input.shape[0])):
                # print('weight ', j, ' times Input ', int((offset+j*dilation)/(z_padding+1)))
                k = 0
                while k < layer_depth:
                    d = 0
                    while d < input.shape[1]:
                        
                        output[i,k] += weights[j,d,k] * input[int((offset+j*dilation)/(z_padding+1)),d]
                        d += 1
                    k += 1                    
            j += 1
        k = 0
        while k < layer_depth:
            output[i,k] += bias[k]*idn
            k += 1
        i += 1

    return output



@njit
def trans_convolution(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a):
    output = np.zeros((input.shape[0]*(z_padding+1),layer_depth))
    i = 0
    while i < output.shape[0]:    
        # print('Ouput ', i, "by")             
        offset = i*strides-padding
        j = 0
        while j < kernel:
            if((offset+(j)*dilation)/(z_padding+1) in range(input.shape[0])):
                # print("weight ", j, " times input ", int((offset+j*dilation)/(z_padding+1)))
                k = 0
                while k < layer_depth:
                    d = 0
                    while d < input.shape[1]:
                        
                        output[i,k] += weights[j,d,k] * input[int((offset+j*dilation)/(z_padding+1)),d]
                        d += 1
                    k += 1                    
            j += 1
        k = 0
        while k < layer_depth:
            output[i,k] += bias[k]
            k += 1
        i += 1

    return output

def trans_convolution_ckks(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a):
    output = np.zeros((input.shape[0]*(z_padding+1),layer_depth),dtype=object)
    i = 0
    while i < output.shape[0]:    
        # print('Ouput ', i, "by")             
        offset = i*strides-padding
        j = 0
        while j < kernel:
            if((offset+(j)*dilation)/(z_padding+1) in range(input.shape[0])):
                # print("weight ", j, " times input ", int((offset+j*dilation)/(z_padding+1)))
                k = 0
                while k < layer_depth:
                    d = 0
                    while d < input.shape[1]:
                        
                        output[i,k] += weights[j,d,k] * input[int((offset+j*dilation)/(z_padding+1)),d]
                        d += 1
                    k += 1                    
            j += 1
        k = 0
        while k < layer_depth:
            output[i,k] += bias[k]
            k += 1
        i += 1

    return output

@njit
def trans_convolution_more(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a):
    idn = np.identity(2)
    output = np.zeros((input.shape[0]*(z_padding+1),layer_depth,2,2))
    i = 0
    while i < output.shape[0]:    
        # print('Ouput ', i, "by")             
        offset = i*strides-padding
        j = 0
        while j < kernel:
            if((offset+(j)*dilation)/(z_padding+1) in range(input.shape[0])):
                # print("weight ", j, " times input ", int((offset+j*dilation)/(z_padding+1)))
                k = 0
                while k < layer_depth:
                    d = 0
                    while d < input.shape[1]:
                        
                        output[i,k] += weights[j,d,k] * input[int((offset+j*dilation)/(z_padding+1)),d]
                        d += 1
                    k += 1                    
            j += 1
        k = 0
        while k < layer_depth:
            output[i,k] += idn*bias[k]
            k += 1
        i += 1

    return output

@njit
def compute_dWeights(input, weights, output_error,  kernel, layer_depth, strides, dilation, z_padding, padding):
    dWeights = np.zeros((kernel, input.shape[1], layer_depth))
    i = 0
    while i < kernel:
        offset, j = i*dilation-padding, 0
        while j < output_error.shape[0]:
            if((offset + strides*j)/(z_padding+1) in range(input.shape[0])):
                k = 0
                while k < layer_depth:
                    d = 0
                    while d < input.shape[1]:
                        dWeights[i,d,k] += output_error[j,k] * input[int((offset+strides*j)/(z_padding+1)),d]
                        d+=1
                    k+=1
            j+=1
        i+=1
    return dWeights

@njit
def compute_dBias(layer_depth, output_error):
    dBias = np.zeros(layer_depth)
    k=0
    while k < layer_depth:
        dBias[k] = layer_depth * np.sum(output_error[:,k])
        k+=1
    return dBias

@njit
def compute_in_error(input_shape, weights, output_error, kernel, layer_depth, strides, dilation, padding, z_padding):
    in_error = np.zeros(input_shape)
    i = 0
    while i < input_shape[0]:
        offset, j = (z_padding+1)*i+padding, 0 
        while j < kernel:
            if((offset-dilation*j)/strides in range(output_error.shape[0])):
                k = 0
                while k < layer_depth:
                    d = 0
                    while d < input_shape[1]:
                        in_error[i,d] += output_error[int((offset-dilation*j)/strides),k] * weights[j,d,k]
                        d+=1
                    k += 1
            j+=1
        i+=1

    return in_error
@njit
def matmul(x,y):
    r = np.zeros((x.shape[0], y.shape[1]))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            for k in range(x.shape[1]):
                r[i,j] += x[i,k]*y[k,j]
    return r