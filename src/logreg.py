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

# # BUILDING NETWORK

net = Network()

fc = FCLayer(input_size, output_size)
a = ActivationLayer(activation=sigmoid, activation_prime=sigmoid_prime)



# context = ts.context(
#             ts.SCHEME_TYPE.CKKS,
#             poly_modulus_degree=8192,
#             coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]         
#           )
# context.generate_galois_keys()
# context.global_scale = 2**21

# more = MoreScheme(40000)
scale = 100

context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192, plain_modulus=1032193)
context.generate_galois_keys()

fc.scale_params_bfv(scale)
# fc.encrypt_params_ckks(context)
# fc.encrypt_params_more(more)

input = x_train[0]


input_bfv = (input*10).astype(int)
print(input_bfv)


i_enc = ts.bfv_vector(context, input_bfv[0])

output = fc.forward_propagation(input)
output_bfv = fc.forward_propagation_bfv(i_enc)

print(output)
for i in range(output_bfv[0][0].shape[0]):

    print(output_bfv[0][0][i].decrypt()[0])




# Encrypt Input

# i_ckks = ts.ckks_vector(context, input[0])

# i_ckks = np.zeros(input.shape, dtype=object)
# for i in range(input.shape[0]):
#     for j in range(input.shape[1]):
#         i_ckks[i,j] = ts.ckks_vector(context, [input[i,j]])

# i_more = np.zeros(input.shape, dtype=object)
# for i in range(input.shape[0]):
#     for j in range(input.shape[1]):
#         i_more[i,j] = more.encrypt(input[i,j])

# Encrypt Output

# o_ckks = np.zeros(output.shape, dtype=object)
# for i in range(output.shape[0]):
#     for j in range(output.shape[1]):
#         o_ckks[i,j] = ts.ckks_vector(context, [output[i,j]])
# o_more = np.zeros(output.shape, dtype=object)
# for i in range(output.shape[0]):
#     for j in range(output.shape[1]):
#         o_more[i,j] = more.encrypt(output[i,j])

#Forward and Backward Propagation with unencrypted Weights



# output_more = fc.forward_propagation_more_encrypted(i_more)
# input_error_more, weights_error_more = fc.backward_propagation_more(o_more, 1)

# output_plain = fc.forward_propagation(input)
# input_error, weights_error = fc.backward_propagation(output, 1)

# output_ckks_enc = fc.forward_propagation_ckks_encrypted(i_ckks)
# i_e_ckks, w_e_ckks = fc.backward_propagation_ckks(o_ckks, 1)

# print(output_plain)
# print(output_ckks_enc[0,0].decrypt())

# print(input_error)
# for i in range(i_e_ckks.shape[1]):
#     print(i_e_ckks[0][i].decrypt())

# print(weights_error)
# for i in range(w_e_ckks.shape[0]):
#     print(w_e_ckks[i][0].decrypt())





# output_more_enc = fc.forward_propagation_more_encrypted(i_more)
# i_e_more, w_e_more = fc.backward_propagation_more(o_more, 1)

# output_more = fc.forward_propagation_more(i_more)
# output_ckks = fc.forward_propagation_ckks(i_ckks)





# print('Plain Output: ',output)
# print('CKKS Output /w plain weights: ', output_ckks.decrypt())
# print('More Output /w plain weights: ', more.decrypt(output_more[0][0]))

# print('CKKS Output /w encrypted weights: ', output_ckks_enc[0,0].decrypt())
# print('More Output /w encrypted weights', more.decrypt(output_more_enc[0][0]))

# print('Plain Errors: ',input_error, weights_error)
# print('CKKS errors: ', i_e_ckks[0][0].decrypt(), w_e_ckks[0][0].decrypt())
# print(i_e_more.shape, w_e_more.shape)
# print('More Output: ')
# for x in i_e_more[0]:
#   print(more.decrypt(x))

# f(w_e_more[:,0])
