

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_functions import tanh, tanh_prime, tanh_more, tanh_prime_more
from loss_functions import mse, mse_prime, bce, bce_prime, mse_prime_more

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

Network
net = Network()
net.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh_more, tanh_prime_more))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh_more, tanh_prime_more))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh_more, tanh_prime_more))
# # train on 1000 samples
# net.use(mse, mse_prime)
# print(x_train.shape)
# net.fit(x_train, y_train, epochs=10, learning_rate=0.1, batch_size = 8)

# net.save("mnist_tanh")

# net = Network()
# net.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
# net.add(ActivationLayer(sigmoid, sigmoid_prime))
# net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
# net.add(ActivationLayer(sigmoid, sigmoid_prime))
# net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
# net.add(ActivationLayer(sigmoid, sigmoid_prime))

# net.use(bce, bce_prime)

# net.fit(x_train, y_train, epochs=10, learning_rate=0.1, batch_size = 8)

# net.save("mnist_sigmoid")

# net = Network()
# net.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
# net.add(ActivationLayer(square, square_prime))
# net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
# net.add(ActivationLayer(square, square_prime))
# net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
# net.add(ActivationLayer(sigmoid, sigmoid_prime))

net.use(mse, mse_prime_more)

net.fit_more(x_train[0:10000], y_train[0:10000], epochs=20, learning_rate=0.1, batch_size = 1)

net.save("mnist_square")
