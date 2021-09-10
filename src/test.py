from scipy import signal
import numpy as np
import time

from conv1D_layer import convolution, convolution_cuda

input_length = 75000
input_depth = 16
kernel = 3
layer_depth = 32
strides = 1
dilation = 1
padding = 0
z_padding = 0
a = 0


input = np.random.rand(input_length,input_depth)
weights = np.random.rand(kernel, input_depth, layer_depth)
bias = np.random.rand(layer_depth)

output2 = convolution_cuda(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a, threads_per_block = 32)


time1 = time.time()
output2 = convolution_cuda(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a, threads_per_block = 32)
time2 = time.time()
output2 = convolution_cuda(input, weights, bias,  kernel, layer_depth, strides, dilation, z_padding, padding, a, threads_per_block = 128)
time3 = time.time()

print(time2-time1)
print(time3-time2)

