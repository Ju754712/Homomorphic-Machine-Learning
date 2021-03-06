

from tensorflow.python.keras.backend import dtype
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from schemes.more import MoreScheme
from activation_functions import sigmoid, sigmoid_prime, sigmoid_approx, sigmoid_approx_prime, sigmoid_more, sigmoid_prime_more, square, square_prime, square_more, square_prime_more
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




more = MoreScheme(2)

# Build Networks

net_sigmoid = Network()
net_sigmoid.load("src/params/mnist_sigmoid")


net_sigmoid_more = Network()
net_sigmoid_more.load("src/params/mnist_sigmoid_more")




for i in range(3):
    net_sigmoid_more.layers[2*i+1] = ActivationLayer(sigmoid, sigmoid_prime)

# Encrypt Input Data


# Predict on Sigmoid Network

time1 = time.time()
output = net_sigmoid.predict(x_test)
time2 = time.time()
output_more = net_sigmoid_more.predict(x_test)
time3 = time.time()
# print("Plain Sigmoid Processing:", time2-time1)
# print("More Sigmoid Processing:", time3-time2)


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

print("Sigmoid Plain: ")
print("accuracy: ", accuracy/len(output))
print("Correct: ", correct, ", incorrect: ", incorrect)
accuracy = 0
correct = 0
incorrect = 0
for i in range(len(output_more)):
    true_value = np.argmax(y_test[i])
    pred_value = np.argmax(output_more[i][0])
    accuracy += mse(y_test[i], output_more[i][0])
    if true_value == pred_value:
        correct +=1
    else: 
        incorrect +=1
print("Sigmoid More")
print("accuracy: ", accuracy/len(output_more))
print("Correct: ", correct, ", incorrect: ", incorrect)


# # Predict Approx

# time1 = time.time()
# output = net_sigmoid_approx.predict(x_test)
# time2 = time.time()
# # output_enc = net_sigmoid_approx_more.predict_more(x_test_enc)
# time3 = time.time()
# print("Plain Approx Processing:", time2-time1)
# print("More Approx Processing:", time3-time2)
        


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
# print("Approx Plain")
# print("accuracy: ", accuracy/len(output))
# print("Correct: ", correct, ", incorrect: ", incorrect)


# print("Decrypting Output")
# output_more= []
# for i in range(len(output_enc)):
#     dec = np.zeros((output_enc[i].shape[0], output_enc[i].shape[1]))
#     for j in range(output_enc[i].shape[0]):
#         for k in range(output_enc[i].shape[1]):
#             dec[j,k] = more.decrypt(output_enc[i][j,k])
#     output_more.append(dec)


# accuracy = 0
# correct = 0
# incorrect = 0
# for i in range(len(output_more)):
#     true_value = np.argmax(y_test[i])
#     pred_value = np.argmax(output_more[i][0])
#     accuracy += mse(y_test[i], output_more[i][0])
#     if true_value == pred_value:
#         correct +=1
#     else: 
#         incorrect +=1
# print("Approx More")
# print("accuracy: ", accuracy/len(output_more))
# print("Correct: ", correct, ", incorrect: ", incorrect)
