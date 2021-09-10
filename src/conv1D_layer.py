from layer import Layer
from scipy import signal
import numpy as np
from numba import njit, cuda
from math import floor, ceil 
import time

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

    # returns output for a given input
    def forward_propagation(self, input):
        self.input = input
        self.output = convolution(input = self.input, weights = self.weights, bias = self.bias, layer_depth = self.layer_depth, kernel = self.kernel, strides = self.strides, dilation = 1, z_padding = 0, padding = self.padding, a = 0) 
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
        if(padding == 'same'):
            self.padding = floor(kernel/2)
        elif(padding == 'valid'):
            self.padding = 0
        else: 
            self.padding = padding
        self.p = kernel-self.padding-1
        self.z_padding = strides - 1
        self.weights = np.random.rand(kernel, self.input_shape[1], layer_depth) - 0.5
        self.bias = np.random.rand(layer_depth) - 0.5


    # returns output for a given input
    def forward_propagation(self, input):
        self.input = input
        self.output= convolution(input = self.input, weights = self.weights, bias = self.bias,  kernel = self.kernel, layer_depth = self.layer_depth, strides = 1, dilation = 1, z_padding = self.z_padding, padding = self.p, a = self.a)
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
    input_length, input_depth = input.shape[0], input.shape[1]
    output = np.zeros((floor((input_length+2*padding+(input_length-1)*z_padding+a-(kernel+(kernel-1)*(dilation-1)))/strides)+1,layer_depth))
    i = 0
    while i < output.shape[0]:                   
        offset = i*strides-padding  
        j = 0
        while j < kernel:
            if((offset+j*dilation)/(z_padding+1) in range(input.shape[0])):
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

@cuda.jit
def convolution_kernel(input, weights, bias, output, kernel, layer_depth, strides, dilation, z_padding, padding):
    # s_input = cuda.shared.array(shape=TPB)
    # s_weights = cuda.shared.array(shape=kernel.shape)
    # s_bias = cuda.shared.array(shape=bias.shape)

    #Absolute postion of thread in grid
    i = cuda.threadIdx.x
    d = cuda.blockIdx.x
    k = cuda.blockIdx.y

    if i >= output.shape[0]:
        # Quit if x is outside of of valid ouput boundary
        return
    offset = i*strides-padding
    j = 0
    while j < kernel:
        if((offset+j*dilation)/(z_padding+1) < input.shape[0] and (offset+j*dilation)%(z_padding+1) == 0): #in range(input.shape[0])
            tmp = weights[j,d,k] * input[int((offset+j*dilation)/(z_padding+1)),d]
            output[i,k] += tmp               
        j +=1
    output[i,k] += bias[k]


def convolution_cuda(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a):
    input_length, input_depth = input.shape[0], input.shape[1]
    output_length = floor((input_length+2*padding+(input_length-1)*z_padding+a-(kernel+(kernel-1)*(dilation-1)))/strides)+1
    output = np.zeros((output_length, layer_depth))

    input_global_mem = cuda.to_device(input)
    weights_global_mem = cuda.to_device(weights)
    bias_global_mem = cuda.to_device(bias)

    output_global_mem = cuda.to_device(output)

    tpb = output_length
    bpg =  (input_depth, layer_depth)
    convolution_kernel[bpg,tpb](input_global_mem, weights_global_mem, bias_global_mem, output_global_mem, kernel, layer_depth, strides, dilation, z_padding, padding)
    output = output_global_mem.copy_to_host()
    
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
