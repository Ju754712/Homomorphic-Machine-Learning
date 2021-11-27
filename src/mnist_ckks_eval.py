

from tensorflow.python.keras.backend import dtype
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from schemes.more import MoreScheme
from activation_functions import sigmoid_approx_ckks
from loss_functions import mse

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


x_test = x_test[0:2]
y_test = y_test[0:2]


# Build Networks

net_sigmoid_approx = Network()
net_sigmoid_approx.load("src/params/mnist_sigmoid_approx")

net_sigmoid_approx_ckks = Network()
net_sigmoid_approx_ckks.load("src/params/mnist_sigmoid_approx")

for i in range(3):
    net_sigmoid_approx_ckks.layers[2*i+1] = ActivationLayer(sigmoid_approx_ckks, sigmoid_approx_ckks)

import tenseal as ts


context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 21, 21, 40]
            
          )
context.generate_galois_keys()
context.global_scale = 2**21

# Encrypt Input Data

print("Encrypting Input")
x_test_enc = np.zeros((x_test.shape[0]),dtype=object)
for i in range(x_test.shape[0]):
    x_test_enc[i]  = ts.CKKSVector(context, x_test[i].reshape(784))

oe1 = x_test_enc[0].matmul(net_sigmoid_approx_ckks.layers[0].weights)+net_sigmoid_approx_ckks.layers[0].bias.reshape(100)
oe2 = oe1.polyval([0.5, 0.197, 0, -0.004])
oe3 = oe2.matmul(net_sigmoid_approx_ckks.layers[2].weights)+net_sigmoid_approx_ckks.layers[2].bias.reshape(50)
oe4 = oe3.polyval([0.5, 0.197, 0, -0.004])
oe5 = oe4.matmul(net_sigmoid_approx_ckks.layers[4].weights)+net_sigmoid_approx_ckks.layers[4].bias.reshape(10)
oe6 = oe5.polyval([0.5, 0.197, 0, -0.004])

o1 = net_sigmoid_approx.layers[0].forward_propagation(x_test[0])
o2 = net_sigmoid_approx.layers[1].forward_propagation(o1)
o3 = net_sigmoid_approx.layers[2].forward_propagation(o2)
o4 = net_sigmoid_approx.layers[3].forward_propagation(o3)
o5 = net_sigmoid_approx.layers[4].forward_propagation(o4)
o6 = net_sigmoid_approx.layers[5].forward_propagation(o5)

oc1 = oe6.decrypt()
print(o6)
print(oc1)