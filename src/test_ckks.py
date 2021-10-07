import tenseal as ts
import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_functions import tanh, tanh_prime, sigmoid, sigmoid_prime, sigmoid_more, sigmoid_prime_more, sigmoid_prime_ckks, sigmoid_ckks
from loss_functions import mse, mse_prime, bce_prime

from schemes.more import MoreScheme
from scipy.linalg import expm

# Implement FCLayer for BFV and CKKS
# Test runtime and precision of plaintext Logistic Regression vs. noise-based integer scheme (BFV), CKKS and noise-free (MORE) 
    # Store measurement directly in .csv
# Deduct which schemes are able to adapt training/evaluation of simple NN for MNIST-classification
# Implement and test autoencoder 

# Setup TenSEAL context
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
            #coeff_mod_bit_sizes=[60, 40, 40, 60]
            
          )
context.generate_galois_keys()
context.global_scale = 2**21

# v1 = [0, 1, 2, 3, 4]
# v2 = [4, 3, 2, 1, 0]

# enc_v1 = ts.ckks_vector(context, v1)
# enc_v2 = ts.ckks_vector(context, v2)

# result = enc_v1 + enc_v2
# print(result.decrypt()) # ~ [4, 4, 4, 4, 4]


# result = enc_v1.dot(enc_v2)
# print(result.decrypt()) # ~ [10]


# matrix = [
#   [73, 0.5, 8],
#   [81, -5, 66],
#   [-100, -78, -2],
#   [0, 9, 17],
#   [69, 11 , 10],
# ]

# result = enc_v1.matmul(matrix)
# print(result.decrypt()) # ~ [157, -90, 153]

# result = enc_v1 * 0.5
# print(result.decrypt()) # ~ [0, 0.5, 1, 1.5, 2]  # Plain multiplication works with rationals

# result = enc_v1 + [1, 1, 1, 1, 1] 
# print(result.decrypt()) # ~ [1, 2, 3, 4, 5] # Plain addition work

# result = enc_v1 * enc_v2
# print(result.decrypt()) # ~ [0, 3, 4, 3, 0]
input_size = 4
output_size = 1

# print(sigmoid(np.array([1,2])))
# print(sigmoid_ckks(ts.ckks_vector(context, [1,2])).decrypt())

input = np.array([[0],[1],[2],[3]])
weights = np.random.rand(input_size, output_size) - 0.5
bias = np.random.rand(output_size) -0.5

input_vector = ts.ckks_vector(context,[0,1,2,3])
enc_w = ts.ckks_vector(context, weights[:,0])

# output = input_vector.matmul(weights) + bias
# print(output.decrypt())
# print(sigmoid_ckks(output).decrypt())

# output = np.dot(input, weights) + bias
# print(output)
# print(sigmoid(output))

output_error = [1]
enc_o = ts.ckks_vector(context, output_error)
input_error = np.dot(output_error, weights.T)
print(input_error)
input_error = enc_o * enc_w
print(input_error.decrypt())

weights_error = np.dot(input, output_error)
print(weights_error)
weights_error = input_vector*enc_o
print(weights_error.decrypt())


    # The symmetric encryption creates smaller contexts than the public key ones.
    # Decreasing the length of the coefficient modulus decreases the size of the context but also the depth of available multiplications.
    # Decreasing the coefficient modulus sizes reduces the context size, but impacts the precision as well (for CKKS).
    # Galois keys increase the context size only for public contexts (without the secret key). Send them only when you need to perform ciphertext rotations on the other end.
    # Relinearization keys increase the context size only for public contexts. Send them only when you need to perform multiplications on ciphertexts on the other end.
    # When we send the secret key, the Relinearization/Galois key can be regenerated on the other end without sending them.
