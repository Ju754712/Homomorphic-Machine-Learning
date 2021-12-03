import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer
from dropout_layer import DropoutLayer
from activation_functions import tanh, tanh_prime, relu, relu_prime, tanh_more, relu_more
from loss_functions import mse, mse_prime
from schemes.more import MoreScheme
import csv
import time

from keras.datasets import mnist
from keras.utils import np_utils
from progress.bar import Bar

more = MoreScheme(2)

PATH = "./src/data/train.npy"
data = np.load(PATH, mmap_mode='r')
x_test = data[0:100]

print(x_test.shape)
time1 = time.time()
enc = more.encrypt_array(x_test)
time2 = time.time()

print(time2-time1)
time1 = time.time()
dec = more.decrypt_array(enc)
time2 = time.time()
print(time2-time1)

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


x_test = x_test[0:100]
print(x_test.shape)

time1 = time.time()
enc = more.encrypt_array(x_test)
time2 =time.time()
dec = more.decrypt_array(enc)
time3 = time.time()

print(time2-time1)
print(time3-time2)