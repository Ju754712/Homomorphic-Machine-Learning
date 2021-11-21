

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


x_train = x_train[0:2]
y_train = y_train[0:2]

x_test = x_test[0:100]
y_test = y_test[0:100]

more = MoreScheme(2)


net = Network()
net.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh_more, tanh_prime_more))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh_more, tanh_prime_more))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh_more, tanh_prime_more))
# # train on 1000 samples

print(net.layers[2].weights)

for i in range(3):
    net.layers[2*i].encrypt_params_more(more)

print(y_train.shape)
print("Encrypting Input")
x_train_enc = np.zeros((x_train.shape[0],x_train.shape[1], x_train.shape[2],2,2))
for i in range(x_train.shape[0]):
    for j in range(x_train.shape[1]):
        for k in range(x_train.shape[2]):
            x_train_enc[i,j,k] = more.encrypt(x_train[i,j,k])
print("Encrypting Output")
y_train_enc = np.zeros((y_train.shape[0],y_train.shape[1],2,2))
for i in range(y_train.shape[0]):
    for j in range(y_train.shape[1]):
        y_train_enc[i,j] = more.encrypt(y_train[i,j])

net.use(mse, mse_prime_more)
print("Training")
net.fit_more(x_train_enc, y_train_enc, epochs=20, learning_rate=0.1, more=more, batch_size = 1)
for i in range(3):
    net.layers[2*i].decrypt_params_more(more)
net.save("mnist_square")
print(net.layers[2].weights)