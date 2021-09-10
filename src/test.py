from scipy import signal
import numpy as np
import time

from conv1D_layer import convolution, convolution_cuda

input_length = 10000
input_depth = 4
kernel = 3
layer_depth = 8
strides = 1
dilation = 1
padding = 0
z_padding = 0
a = 0


input = np.random.rand(input_length,input_depth)
weights = np.random.rand(kernel, input_depth, layer_depth)
bias = np.random.rand(layer_depth)

output = convolution(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a)
output2 = convolution_cuda(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a)


time1 = time.time()
output = convolution(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a)
time2 = time.time()
output2 = convolution_cuda(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a)
time3 = time.time()

print(time2-time1)
print(time3-time2)

print(output[0], output2[0])
