

from tensorflow.python.keras.backend import dtype
from tensorflow.python.keras.layers.core import Activation
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from schemes.more import MoreScheme
from activation_functions import sigmoid_approx_ckks, square, square_ckks, square_prime
from loss_functions import mse
from progress.bar import Bar

from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

import time



# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)


x_test = x_test[300:400]
y_test = y_test[300:400]


# Build Networks

net_sigmoid_approx = Network()
net_sigmoid_approx.load("src/params/mnist_sigmoid_approx")

net_sigmoid_approx_ckks = Network()
net_sigmoid_approx_ckks.load("src/params/mnist_sigmoid_approx")

for i in range(3):
    net_sigmoid_approx.layers[2*i+1] = ActivationLayer(square, square_prime)
    net_sigmoid_approx_ckks.layers[2*i+1] = ActivationLayer(square_ckks, square_prime)

import tenseal as ts

bit_scale = 40
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=2*8192,
            coeff_mod_bit_sizes = [60, bit_scale, bit_scale, bit_scale, bit_scale, bit_scale, bit_scale, 60]
            
          )
context.generate_galois_keys()
context.global_scale = 2**bit_scale

# Encrypt Input Data
# print(x_test.shape)
# x_test_enc = np.zeros(x_test.shape, dtype=object)
# for i in range(x_test.shape[0]):
#     for j in range(x_test.shape[1]):
#         for k in range(x_test_enc.shape[2]):
#             x_test_enc[i,j,k] = ts.CKKSVector(context, [x_test[i,j,k]])


# print("Encrypting Input")
# # time1 = time.time()
# # output = net_sigmoid_approx.predict(x_test)
# # time2 = time.time()
# # output_ckks = net_sigmoid_approx_ckks.predict_ckks(x_test_enc)
# # time3 = time.time()
# # time_plain = time2-time1
# # time_ckks = time3-time2

# o1 = net_sigmoid_approx.layers[0].forward_propagation(x_test[0])
# oe1 = net_sigmoid_approx_ckks.layers[0].forward_propagation_ckks(x_test_enc[0])
# o2 = net_sigmoid_approx.layers[1].forward_propagation(o1)
# o = net_sigmoid_approx_ckks.layers[0].forward_propagation_ckks(oe1)

# o_dec = np.zeros(o.shape)
# for i in range(o.shape[0]):
#     for j in range(o.shape[1]):
#         o_dec[i,j] = o[i,j].decrypt()[0]

# print(o2[0,0:10])
# print(o_dec[0,0:10])


# output_ckks_dec = []
# for i in range(len(output_ckks)):
#     dec = np.zeros(output_ckks[i].shape)
#     for j in range(output_ckks[i].shape[0]):
#         for k in range(output_ckks[i].shape[1]):
#             d = output_ckks[i][j,k].decrypt()
#             print(d)
#             dec[j,k] = d[0]
#     output_ckks_dec.append(dec)


output_ckks_packed = []
time_enc_packed = 0
time_ckks_packed = 0
time_dec_packed = 0
bar = Bar("Processing CKKS...", max = x_test.shape[0])
for i in range(x_test.shape[0]):
    bar.next()
    time1 = time.time()
    x_test_enc = ts.CKKSVector(context, x_test[i].reshape(784))
    time2 = time.time()
    time_enc_packed += time2-time1
    time1 = time.time()
    oe1 = x_test_enc.matmul(net_sigmoid_approx_ckks.layers[0].weights)+net_sigmoid_approx_ckks.layers[0].bias.reshape(100)
    oe2 = oe1.square()+oe1
    oe3 = oe2.matmul(net_sigmoid_approx_ckks.layers[2].weights)+net_sigmoid_approx_ckks.layers[2].bias.reshape(50)
    oe4 = oe3.square()+oe3
    oe5 = oe4.matmul(net_sigmoid_approx_ckks.layers[4].weights)+net_sigmoid_approx_ckks.layers[4].bias.reshape(10)
    oe6 = oe5.square()+oe5
    time2 = time.time()
    time_ckks_packed += time2-time1
    time1 = time.time()
    o = oe6.decrypt()
    time2 = time.time()
    time_dec_packed += time2-time1
    output_ckks_packed.append(o)
bar.finish()


time1 = time.time()
# x_test_enc = np.zeros(x_test.shape, dtype=object)
# for i in range(x_test.shape[0]):
#     for j in range(x_test.shape[1]):
#         for k in range(x_test_enc.shape[2]):
#             x_test_enc[i,j,k] = ts.CKKSVector(context, [x_test[i,j,k]])
time2 = time.time()

time_enc = time2-time1

time1 = time.time()
output = net_sigmoid_approx.predict(x_test)
time2 = time.time()
# output_ckks = net_sigmoid_approx_ckks.predict_ckks(x_test_enc)
time3 = time.time()
time_plain = time2-time1
time_ckks = time3-time2

output_ckks_dec = []
# time1 = time.time()
# for i in range(len(output_ckks)):
#     dec = np.zeros(output_ckks[i].shape)
#     for j in range(output_ckks[i].shape[0]):
#         for k in range(output_ckks[i].shape[1]):
#             d = output_ckks[i][j,k].decrypt()
#             print(d)
#             dec[j,k] = d[0]
#     output_ckks_dec.append(dec)
time2 = time.time()
time_dec = time2-time1

print(output[0].shape)
print(len(output_ckks_packed[0]))

accuracy = 0
accuracy_ckks=0
accuracy_ckks_packed = 0
correct = 0
correct_ckks = 0
correct_ckks_packed = 0
for i in range(len(output)):
    y_true = np.argmax(y_test[i])
    y_pred = np.argmax(output[i][0])
    # y_pred_ckks = np.argmax(output_ckks_dec[i])
    y_pred_ckks_packed = np.argmax(output_ckks_packed[i])
    accuracy += mse(output[i],y_test[i])
    # accuracy_ckks += mse(output_ckks_dec[i],y_test[i])
    print(output[i])
    # print(output_ckks_dec[i])
    print(output_ckks_packed[i])
    print(y_test[i])
    accuracy_ckks_packed += mse(output_ckks_packed[i], y_test[i][0])

    if y_pred == y_true:
        correct +=1
    else:
        print("Plain predicted ",y_pred, ", but real value was ",y_true)
    # if y_pred_ckks == y_true:
    #     correct_ckks += 1
    if y_pred_ckks_packed == y_true:
        correct_ckks_packed += 1
    else:
        print("Plain predicted ",y_pred_ckks_packed, ", but real value was ",y_true)
accuracy = accuracy/len(output)
accuracy_ckks = accuracy_ckks/len(output)
accuracy_ckks_packed = accuracy_ckks_packed/len(output)

print("Time Plain: ", time_plain)

print("Time Encryption: ", time_enc)
print("Time CKKS: ", time_ckks)
print("Time Decryption: ", time_dec)

print("Time Encryption Packed: ", time_enc_packed)
print("Time CKKS Packed: ", time_ckks_packed)
print("Time Decryption Packed: ", time_dec_packed)

print("Accuracy Plain: ", accuracy)
print("Accuracy CKKS: ", accuracy_ckks)
print("Accuracy CKKS Packed: ", accuracy_ckks_packed)
print("Correct Plain: ", correct)
print("Correct CKKS: ", correct_ckks)
print("Correct CKKS Packed: ", correct_ckks_packed)



