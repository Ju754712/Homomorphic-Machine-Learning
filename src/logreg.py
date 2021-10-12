import matplotlib.pyplot as plt
from data_setup import heart_disease_data, titanic_data, random_data
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_functions import sigmoid, sigmoid_prime, tanh, tanh_prime, sigmoid_ckks, sigmoid_prime_ckks
from loss_functions import bce, bce_prime, mse, mse_prime
import tenseal as ts
from schemes.more import MoreScheme
import numpy as np
x_train, y_train, x_test, y_test = heart_disease_data()



input_size, output_size = x_train.shape[2], y_train.shape[2]

# # BUILDING NETWORK

net = Network()

net.add(FCLayer(input_size, output_size))
net.add(ActivationLayer(activation=sigmoid, activation_prime = sigmoid_prime))

#net.use(bce, bce_prime)
net.use(bce, bce_prime)
net.fit(x_train, y_train, epochs=1, learning_rate=0.5,batch_size=4, shuffle=True, adaptive=0.01)
correct = 0
for i in range(x_test.shape[0]):
    out = net.predict(x_test[i])
    if abs(out[0][0,0]-y_test[i,0,0]) < 0.5:
        correct +=1
print("Plain Accuracy:", correct/x_test.shape[0])



context_ckks = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]         
          )
context_ckks.generate_galois_keys()
context_ckks.global_scale = 2**21
net.layers[0].encrypt_params_ckks(context_ckks)
a = ActivationLayer(activation=sigmoid_ckks, activation_prime=sigmoid_prime_ckks)
correct_ckks = 0


for i in range(x_test.shape[0]):
    x_in = ts.ckks_vector(context_ckks, x_test[i,0])
    out = net.layers[0].forward_propagation_ckks(x_in)
    out = a.forward_propagation(out)
    if abs(out.decrypt()-y_test[i,0,0]) < 0.5:
        correct_ckks +=1

print(correct_ckks/x_test.shape[0])    



more = MoreScheme(1000)

net.layers[0].encrypt_params_more(more)

a = ActivationLayer(activation=sigmoid_more, activation_prime= sigmoid_prime_more)

correct_more = 0


for i in range(x_test.shape[0]):
    input_more = np.zeros(x_train[i].shape, dtype=object)
    for k in range(x_train[i].shape[0]):
        for m in range(x_train[j].shape[1]):
            input_more[k,m] = more.encrypt(x_train[j,k,m])

    out = net.layers[0].forward_propagation_more(input_more)
    out = a.forward_propagation(out)
    if abs(more.decrypt(out)-y_test[i,0,0]) < 0.5:
        correct_more +=1

print(correct_more/x_test.shape[0])   
