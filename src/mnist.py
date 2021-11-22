

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
net.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
net.add(ActivationLayer(tanh, tanh_prime))

net.save("test")

net_more = Network()
net_more.load("test")
# # # train on 1000 samples


# print(net.layers[2].weights)

for i in range(3):
    net_more.layers[2*i+1] = ActivationLayer(tanh_more, tanh_prime_more)
    net_more.layers[2*i].encrypt_params_more(more)

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



net.use(mse, mse_prime)
net_more.use(mse, mse_prime_more)

print(net.layers[2].weights[0])
print(net_more.layers[2].weights[0])

net.fit(x_train, y_train, epochs=20, learning_rate = 0.1, batch_size =1)
net_more.fit_more(x_train_enc, y_train_enc, epochs=20, learning_rate=0.1, batch_size = 1, more=more)

for i in range(3):
    net_more.layers[2*i].decrypt_params_more(more)

print(net.layers[2].weights[0])
print(net_more.layers[2].weights[0])
print(net.layers[2].bias[0])
print(net_more.layers[2].bias[0])
# net.save("mnist_square")
# print(net.layers[2].weights)



# print("Before: ", net.layers[0].weights[0])

# output_plain = net.layers[0].forward_propagation(x_train[0])
# error_prime = mse_prime(y_train[0], output_plain)
# output_error, weights_error = net.layers[0].backward_propagation(error_prime, 0.1)

# print("After Plain: ", net.layers[0].weights[0])

# output_enc = net.layers[0].forward_propagation_more_encrypted(x_train_enc[0])
# error_prime_enc = mse_prime_more(y_train_enc[0], output_enc)
# output_error_enc, weights_error_enc = net.layers[0].backward_propagation_more(error_prime_enc, 0.1)


# output_more = np.zeros((output_enc.shape[0],output_enc.shape[1]))
# for i in range(output_enc.shape[0]):
#     for j in range(output_enc.shape[1]):
#         output_more[i,j] = more.decrypt(output_enc[i,j])

# error_prime_more = np.zeros((error_prime_enc.shape[0],error_prime_enc.shape[1]))
# for i in range(error_prime_enc.shape[0]):
#     for j in range(error_prime_enc.shape[1]):
#         error_prime_more[i,j] = more.decrypt(error_prime_enc[i,j])

# output_error_more = np.zeros((output_error_enc.shape[0],output_error_enc.shape[1]))
# for i in range(output_error_enc.shape[0]):
#     for j in range(output_error_enc.shape[1]):
#         output_error_more[i,j] = more.decrypt(output_error_enc[i,j])

        
# weights_error_more = np.zeros((weights_error_enc.shape[0],weights_error_enc.shape[1]))
# for i in range(weights_error_enc.shape[0]):
#     for j in range(weights_error_enc.shape[1]):
#         weights_error_more[i,j] = more.decrypt(weights_error_enc[i,j])


# net.layers[0].decrypt_params_more(more)


