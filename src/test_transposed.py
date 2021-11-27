import numpy as np
# from tensorflow import keras

# import tensorflow as tf
from activation_layer import ActivationLayer
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer
from activation_functions import tanh, tanh_prime, relu, relu_prime, sigmoid_approx , sigmoid_approx_prime, sigmoid_approx_more, sigmoid_approx_prime_more, sigmoid, sigmoid_prime, sigmoid_more, sigmoid_prime_more
from network import Network 
from tensorflow.keras import layers
from data_setup import random_data
from fc_layer import FCLayer
from schemes.more import MoreScheme

more = MoreScheme(2)

x = np.array([[1]])
x_enc = np.array([[more.encrypt(1)]])

sigmoidLayer = ActivationLayer(sigmoid_approx, sigmoid_approx_prime)
sigmoidMoreLayer = ActivationLayer(sigmoid_approx_more, sigmoid_approx_prime_more)
output = sigmoidLayer.forward_propagation(x)
output_enc = sigmoidLayer.forward_propagation_more(x_enc)
output_more = np.zeros((output_enc.shape[0],output_enc.shape[1]))
for i in range(output_enc.shape[0]):
    for j in range(output_enc.shape[1]):
        output_more[i,j] = more.decrypt(output_enc[i,j])
print(output)
print(output_more)