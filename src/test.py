from scipy import signal
import numpy as np
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer, convolution, compute_dWeights, compute_in_error
from activation_layer import ActivationLayer
from activation_functions import relu, relu_prime
from fc_layer import FCLayer
from dropout_layer import DropoutLayer
import time
input_length = 150000
input_depth = 32
kernel = 7
layer_depth = 16
strides = 1
dilation = 1
z_padding = 0
padding = 0
a=0


# for i in range(10):
#     data = np.random.rand(input_length,input_depth)
#     weights = np.random.rand(kernel,input_depth,layer_depth)
#     bias = np.random.rand(layer_depth)
#     time1 = time.time()
#     output = convolution(input = data, weights = weights, bias = bias, kernel = kernel, layer_depth = layer_depth, strides = strides, dilation = dilation, z_padding = z_padding, padding = padding, a = 0)
#     dWeights = compute_dWeights(input = data, weights = weights, output_error = output, kernel = kernel, layer_depth = layer_depth, strides = strides, dilation = dilation, z_padding = z_padding, padding = padding)
#     time2 = time.time()
#     print(time2-time1)
    

# dWeights = compute_dWeights(input = data, weights = weights, output_error = output, kernel = kernel, layer_depth = layer_depth, strides = strides, dilation = dilation, z_padding = z_padding, padding = padding)
# time1 = time.time()
# dWeights = compute_dWeights(input = data, weights = weights, output_error = output, kernel = kernel, layer_depth = layer_depth, strides = strides, dilation = dilation, z_padding = z_padding, padding = padding)
# in_error = compute_in_error(input_shape = data.shape, weights = weights, output_error = output, kernel = kernel, layer_depth = layer_depth, strides = strides, dilation = dilation, padding = padding, z_padding = z_padding)

# time2 = time.time()

# print(data)
# data_2 = np.zeros((data.shape[0]+3, data.shape[1]))
# data_2[0:data.shape[0],:] = data
# print(data_2)

# layer = Conv1DTransposedLayer(input_shape=data.shape, kernel = kernel, layer_depth=layer_depth, strides=strides, padding=padding, a = a)
# output = layer.forward_propagation(data)
# print(output.shape)

import sys

toolbar_width = 40

# setup toolbar
sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1))

for i in range(toolbar_width):
    time.sleep(0.1) # do real work here
    # update the bar
    sys.stdout.write("-")
    sys.stdout.flush()

sys.stdout.write("]\n") # this ends the progress bar

sys.stdout.write("[%s]" % (" " * toolbar_width))
sys.stdout.flush()
sys.stdout.write("\b" * (toolbar_width+1))

for i in range(toolbar_width):
    time.sleep(0.1) # do real work here
    # update the bar
    sys.stdout.write("-")
    sys.stdout.flush()

sys.stdout.write("]\n") # this ends the progress bar
