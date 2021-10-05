import matplotlib.pyplot as plt
from data_setup import heart_disease_data, titanic_data
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_functions import sigmoid, sigmoid_prime, tanh, tanh_prime
from loss_functions import bce, bce_prime, mse, mse_prime
import tenseal as ts
from schemes.more import MoreScheme
import numpy as np
x_train, y_train, x_test, y_test = heart_disease_data()

input_size, output_size = x_train.shape[2], y_train.shape[2]

print(x_train.shape)

# # BUILDING NETWORK

net = Network()

fc = FCLayer(input_size, output_size)
a = ActivationLayer(activation=sigmoid, activation_prime=sigmoid_prime)

context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]         
          )
context.generate_galois_keys()
context.global_scale = 2**21

more = MoreScheme(400)

fc.encrypt_params_ckks(context)
fc.encrypt_params_more(more)

input = x_train[0]
print(input[0])
i_ckks = ts.ckks_vector(context, input[0])
i_more = np.zeros(input.shape, dtype=object)
for i in range(input.shape[0]):
    for j in range(input.shape[1]):
        i_more[i,j] = more.encrypt(input[i,j])

print(i_more.shape)
output = fc.forward_propagation(input)
output_more = fc.forward_propagation_more_encrypted(i_more)
output_ckks = fc.forward_propagation_ckks(i_ckks)
print(output)
print(more.decrypt(output_more[0,0]))
print(output_ckks.decrypt())
# print(output_ckks)

