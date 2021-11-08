import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_functions import tanh, tanh_prime, sigmoid, sigmoid_prime, square, square_prime,tanh_more, tanh_prime_more
from loss_functions import mse, mse_prime, bce, bce_prime, mse_more, mse_prime_more
from schemes.more import MoreScheme

from keras.datasets import mnist
from keras.utils import np_utils

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

x_train = x_train[0:10]
y_train = y_train[0:10]
#Build network

net = Network()
net.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh_more, tanh_prime_more))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh_more, tanh_prime_more))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh_more, tanh_prime_more))

net.use(mse_more, mse_prime_more)

more = MoreScheme(2)
for i in range(3):
    net.layers[2*i].encrypt_params_more(more)

x_train_more = np.zeros(x_train.shape, dtype=object)
y_train_more = np.zeros(y_train.shape, dtype=object)

for i in range(x_train_more.shape[0]):
    for k in range(x_train_more.shape[1]):
        for m in range(x_train_more.shape[2]):
            x_train_more[i,k,m ]= more.encrypt(x_train[i,k,m])
    for k in range(y_train.shape[1]):
        y_train_more[i,k] = more.encrypt(y_train[i,k])

net.fit_more(x_train_more, y_train_more, epochs=1, learning_rate=1, batch_size = 1)
