import tenseal as ts
import numpy as np

from network import Network
from fc_layer import FCLayer, MORE_FCLayer
from activation_layer import ActivationLayer
from activation_functions import tanh, tanh_prime, sigmoid, sigmoid_prime
from loss_functions import mse, mse_prime, bce_prime

from keras.datasets import mnist
from keras.utils import np_utils

from schemes.more import MoreScheme

# more = MoreScheme(1000)
# more.keygen()
# input_size = 28
# output_size = 10

# fc = FCLayer(input_size, output_size)
# fc_more = MORE_FCLayer(input_size, output_size, more)

# weights = fc.weights
# bias = fc.bias
# for j in range(output_size):
#     for i in range(input_size):
#         fc_more.weights[i,j] = more.encrypt(fc.weights[i,j])
#     fc_more.bias[0,j] = more.encrypt(fc.bias[0,j])

# input = np.random.rand(1,input_size)
# enc_input = np.zeros((1,input_size), dtype = object)
# for i in range(input_size):
#     enc_input[0,i] = more.encrypt(input[0,i])

# o=fc.forward_propagation(input)
# o_enc = fc_more.forward_propagation(enc_input)

# for i in range(output_size):
#     print(o[0,i])
#     print(more.decrypt(o_enc[0,i]))

y_pred = np.array([[1.e-15]])
print(y_pred)
y_true = np.array([[1]])

print(bce_prime(y_true,y_pred))