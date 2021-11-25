

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from schemes.more import MoreScheme
from activation_functions import sigmoid, sigmoid_prime, sigmoid_more, sigmoid_prime_more, sigmoid_approx , sigmoid_approx_prime, sigmoid_approx_more, sigmoid_approx_prime_more
from loss_functions import mse, mse_prime, bce, bce_prime, mse_prime_more

from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import time

EPOCHS = 35
BATCH_SIZE = 1
LEARNING_RATE = 0.1



# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()


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


# x_train = x_train[0:4000]
# y_train = y_train[0:1000]

x_test = x_test[0:10000]
y_test = y_test[0:10000]

more = MoreScheme(200)
time1 = time.time()
print("Encrypting Input")
x_train_enc = np.zeros((x_train.shape[0],x_train.shape[1], x_train.shape[2],2,2))
for i in range(x_train.shape[0]):
    for j in range(x_train.shape[1]):
        for k in range(x_train.shape[2]):
            x_train_enc[i,j,k] = more.encrypt(x_train[i,j,k])
time2 = time.time()
print("Encrypting Output")
y_train_enc = np.zeros((y_train.shape[0],y_train.shape[1],2,2))
for i in range(y_train.shape[0]):
    for j in range(y_train.shape[1]):
        y_train_enc[i,j] = more.encrypt(y_train[i,j])
time3 = time.time()

print("Time for x_train encryption: ", time2-time1)

print("Time for y_train encryption: ", time3-time2)

net_sigmoid = Network()
net_sigmoid.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net_sigmoid.add(ActivationLayer(sigmoid, sigmoid_prime))
net_sigmoid.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net_sigmoid.add(ActivationLayer(sigmoid, sigmoid_prime))
net_sigmoid.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net_sigmoid.add(ActivationLayer(sigmoid, sigmoid_prime))

net_sigmoid.save("test")

net_sigmoid_more = Network()
net_sigmoid_more.load("test")

for i in range(3):
    net_sigmoid_more.layers[2*i+1] = ActivationLayer(sigmoid_more, sigmoid_prime_more)
    net_sigmoid_more.layers[2*i].encrypt_params_more(more)


net_sigmoid.use(mse, mse_prime)
net_sigmoid_more.use(mse, mse_prime_more)

time1 = time.time()
net_sigmoid.fit(x_train, y_train, epochs=EPOCHS, learning_rate = LEARNING_RATE, batch_size =BATCH_SIZE)
time2 = time.time()
net_sigmoid_more.fit_more(x_train_enc, y_train_enc, epochs=EPOCHS, learning_rate = LEARNING_RATE, batch_size =BATCH_SIZE, more=more)
time3 = time.time()

print("Training Sigmoid Plain: ",time2-time1)
print("Training Sigmoid More: ",time3-time2)

for i in range(3):
    net_sigmoid_more.layers[2*i].decrypt_params_more(more)

net_sigmoid.save("src/params/mnist_sigmoid")
net_sigmoid_more.save("src/params/mnist_sigmoid_more")

net_sigmoid = Network()
net_sigmoid.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net_sigmoid.add(ActivationLayer(sigmoid_approx, sigmoid_approx_prime))
net_sigmoid.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net_sigmoid.add(ActivationLayer(sigmoid_approx, sigmoid_approx_prime))
net_sigmoid.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net_sigmoid.add(ActivationLayer(sigmoid_approx, sigmoid_approx_prime))

net_sigmoid.save("test")

net_sigmoid_more = Network()
net_sigmoid_more.load("test")

for i in range(3):
    net_sigmoid_more.layers[2*i+1] = ActivationLayer(sigmoid_approx_more, sigmoid_approx_prime_more)
    net_sigmoid_more.layers[2*i].encrypt_params_more(more)


net_sigmoid.use(mse, mse_prime)
net_sigmoid_more.use(mse, mse_prime_more)

time1 = time.time()
net_sigmoid.fit(x_train, y_train, epochs=EPOCHS, learning_rate = LEARNING_RATE, batch_size =BATCH_SIZE)
time2 = time.time()
net_sigmoid_more.fit_more(x_train_enc, y_train_enc, epochs=EPOCHS, learning_rate = LEARNING_RATE, batch_size =BATCH_SIZE, more=more)
time3 = time.time()

print("Training Sigmoid Plain: ",time2-time1)
print("Training Sigmoid More: ",time3-time2)

for i in range(3):
    net_sigmoid_more.layers[2*i].decrypt_params_more(more)

net_sigmoid.save("src/params/mnist_sigmoid_approx")
net_sigmoid_more.save("src/params/mnist_sigmoid_approx_more")

