

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from schemes.more import MoreScheme
from activation_functions import tanh, tanh_prime, tanh_more, tanh_prime_more
from loss_functions import mse, mse_prime, bce, bce_prime, mse_prime_more

from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

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


x_train = x_train[0:1000]
y_train = y_train[0:1000]

x_test = x_test[0:100]
y_test = y_test[0:100]

more = MoreScheme(2)

net = Network()
net.load("src/params/mnist_more")


for i in range(3):
    net.layers[2*i+1] = ActivationLayer(tanh, tanh_prime)

output = net.predict(x_test[0:3])
print(output)
print(y_test[0:3])
