import numpy as np
# from tensorflow import keras

# import tensorflow as tf
from activation_layer import ActivationLayer
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer
from activation_functions import tanh, tanh_prime, relu, relu_prime, sigmoid_approx , sigmoid_approx_prime, sigmoid_approx_more, sigmoid_approx_prime_more, sigmoid, sigmoid_prime, sigmoid_more, sigmoid_prime_more
from network import Network 
from tensorflow.keras import layers
from data_setup import random_data
from schemes.more import MoreScheme

more = MoreScheme(2)

x = np.array([[1]])
x_enc = np.array([[more.encrypt(1)]])

sigmoidLayer = ActivationLayer(activation=sigmoid, activation_prime=sigmoid_prime)
sigmoidMore = ActivationLayer(activation=sigmoid_more, activation_prime=sigmoid_prime_more)

y = sigmoidLayer.forward_propagation(x)
y_enc = sigmoidMore.forward_propagation_more_encrypted(x_enc)

print(y)
print(more.decrypt(y_enc[0][0]))

y = sigmoidLayer.backward_propagation(x, 1)
y_enc = sigmoidMore.backward_propagation_more(x_enc,1)

print(y)
print(more.decrypt(y_enc[0][0]))

sigmoidLayer = ActivationLayer(activation=sigmoid_approx, activation_prime=sigmoid_approx_prime)
sigmoidMore = ActivationLayer(activation=sigmoid_approx_more, activation_prime=sigmoid_approx_prime_more)

y = sigmoidLayer.forward_propagation(x)
y_enc = sigmoidMore.forward_propagation_more_encrypted(x_enc)

print(y)
print(more.decrypt(y_enc[0][0]))

y = sigmoidLayer.backward_propagation(x, 1)
y_enc = sigmoidMore.backward_propagation_more(x_enc,1)

print(y)
print(more.decrypt(y_enc[0][0]))