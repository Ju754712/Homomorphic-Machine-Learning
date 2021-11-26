

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from schemes.more import MoreScheme
from activation_functions import sigmoid, sigmoid_prime, sigmoid_approx, sigmoid_approx_prime
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


x_train = x_train[0:1000]
y_train = y_train[0:1000]

x_test = x_test[0:10000]
y_test = y_test[0:10000]

more = MoreScheme(2)

net_sigmoid = Network()
net_sigmoid.load("src/params/mnist_sigmoid")


net_sigmoid_approx = Network()
net_sigmoid_approx.load("src/params/mnist_sigmoid_approx")

net_sigmoid_more = Network()
net_sigmoid_more.load("src/params/mnist_sigmoid")


net_sigmoid_approx_more = Network()
net_sigmoid_approx_more.load("src/params/mnist_sigmoid_approx_more")



for i in range(3):
    net_sigmoid_more.layers[2*i+1] = ActivationLayer(sigmoid, sigmoid_prime)
    net_sigmoid_approx_more.layers[2*i+1] = ActivationLayer(sigmoid_approx, sigmoid_approx_prime)
time1 = time.time()
output = net_sigmoid.predict(x_test)
time2 = time.time()
print(time2-time1)
accuracy = 0
correct = 0
incorrect = 0
for i in range(len(output)):
    true_value = np.argmax(y_test[i])
    pred_value = np.argmax(output[i][0])
    accuracy += mse(y_test[i], output[i][0])
    if true_value == pred_value:
        correct +=1
    else: 
        incorrect +=1

print("accuracy: ", accuracy/len(output))
print("Correct: ", correct, ", incorrect: ", incorrect)



# output = net_sigmoid_approx.predict(x_test)

# accuracy = 0
# correct = 0
# incorrect = 0
# for i in range(len(output)):
#     true_value = np.argmax(y_test[i])
#     pred_value = np.argmax(output[i][0])
#     accuracy += mse(y_test[i], output[i][0])
#     if true_value == pred_value:
#         correct +=1
#     else: 
#         incorrect +=1

# print("accuracy: ", accuracy/len(output))
# print("Correct: ", correct, ", incorrect: ", incorrect)

# output = net_sigmoid_more.predict(x_test)

# accuracy = 0
# correct = 0
# incorrect = 0
# for i in range(len(output)):
#     true_value = np.argmax(y_test[i])
#     pred_value = np.argmax(output[i][0])
#     accuracy += mse(y_test[i], output[i][0])
#     if true_value == pred_value:
#         correct +=1
#     else: 
#         incorrect +=1

# print("accuracy: ", accuracy/len(output))
# print("Correct: ", correct, ", incorrect: ", incorrect)


# output = net_sigmoid_approx_more.predict(x_test)

# accuracy = 0
# correct = 0
# incorrect = 0
# for i in range(len(output)):
#     true_value = np.argmax(y_test[i])
#     pred_value = np.argmax(output[i][0])
#     accuracy += mse(y_test[i], output[i][0])
#     if true_value == pred_value:
#         correct +=1
#     else: 
#         incorrect +=1

# print("accuracy: ", accuracy/len(output))
# print("Correct: ", correct, ", incorrect: ", incorrect)