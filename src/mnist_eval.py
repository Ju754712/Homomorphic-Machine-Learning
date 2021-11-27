

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from schemes.more import MoreScheme
from activation_functions import sigmoid, sigmoid_prime, sigmoid_approx, sigmoid_approx_prime, sigmoid_more, sigmoid_prime_more, sigmoid_approx_more, sigmoid_approx_prime_more
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

x_test = x_test[0:2]
y_test = y_test[0:2]

more = MoreScheme(2)

net_sigmoid = Network()
net_sigmoid.load("src/params/mnist_sigmoid")


net_sigmoid_approx = Network()
net_sigmoid_approx.load("src/params/mnist_sigmoid_approx")

net_sigmoid_more = Network()
net_sigmoid_more.load("src/params/mnist_sigmoid")


net_sigmoid_approx_more = Network()
net_sigmoid_approx_more.load("src/params/mnist_sigmoid_approx")



for i in range(3):
    net_sigmoid_more.layers[2*i+1] = ActivationLayer(sigmoid_more, sigmoid_prime_more)
    net_sigmoid_approx_more.layers[2*i+1] = ActivationLayer(sigmoid_approx_more, sigmoid_approx_prime_more)

print("Encrypting Input")
x_test_enc = np.zeros((x_test.shape[0],x_test.shape[1], x_test.shape[2],2,2))
for i in range(x_test.shape[0]):
    for j in range(x_test.shape[1]):
        for k in range(x_test.shape[2]):
            x_test_enc[i,j,k] = more.encrypt(x_test[i,j,k])
time2 = time.time()

print("Encrypting Output")
y_test_enc = np.zeros((y_test.shape[0],y_test.shape[1],2,2))
for i in range(y_test.shape[0]):
    for j in range(y_test.shape[1]):
        y_test_enc[i,j] = more.encrypt(y_test[i,j])
time3 = time.time()


# time1 = time.time()
# output = net_sigmoid.predict(x_test)
# time2 = time.time()
# output_enc = net_sigmoid_more.predict_more(x_test_enc)
# time3 = time.time()
# print("Plain Processing:", time2-time1)
# print("More Processing:", time3-time2)


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

# print("accuracy: ", accuracy/len(output_more))
# print("Correct: ", correct, ", incorrect: ", incorrect)

# time1 = time.time()
# output = net_sigmoid_approx.predict(x_test)
# time2 = time.time()
# output_enc = net_sigmoid_approx_more.predict_more(x_test_enc)
# time3 = time.time()
# print("Plain Processing:", time2-time1)
# print("More Processing:", time3-time2)


o1 = net_sigmoid_approx.layers[0].forward_propagation(x_test[0])
o2 = net_sigmoid_approx.layers[1].forward_propagation(o1)
o3 = net_sigmoid_approx.layers[2].forward_propagation(o2)
o4 = net_sigmoid_approx.layers[3].forward_propagation(o3)
o5 = net_sigmoid_approx.layers[4].forward_propagation(o4)
o6 = net_sigmoid_approx.layers[5].forward_propagation(o5)

om1 = net_sigmoid_approx_more.layers[0].forward_propagation_more(x_test_enc[0])
om2 = net_sigmoid_approx_more.layers[1].forward_propagation_more(om1)
om3 = net_sigmoid_approx_more.layers[2].forward_propagation_more(om2)
om4 = net_sigmoid_approx_more.layers[3].forward_propagation_more(om3)
om5 = net_sigmoid_approx_more.layers[4].forward_propagation_more(om4)
om6 = net_sigmoid_approx_more.layers[5].forward_propagation_more(om5)

print(o6.shape)
print(om6.shape)

ome1 = np.zeros((om1.shape[0],om1.shape[1]))
for i in range(om1.shape[0]):
    for j in range(om1.shape[1]):
        ome1[i,j] = more.decrypt(om1[i,j])
ome2 = np.zeros((om2.shape[0],om2.shape[1]))
for i in range(om2.shape[0]):
    for j in range(om2.shape[1]):
        ome2[i,j] = more.decrypt(om2[i,j])
ome3 = np.zeros((om3.shape[0],om3.shape[1]))
for i in range(om3.shape[0]):
    for j in range(om3.shape[1]):
        ome3[i,j] = more.decrypt(om3[i,j])
ome4 = np.zeros((om4.shape[0],om4.shape[1]))
for i in range(om4.shape[0]):
    for j in range(om4.shape[1]):
        ome4[i,j] = more.decrypt(om4[i,j])
ome5 = np.zeros((om5.shape[0],om5.shape[1]))
for i in range(om5.shape[0]):
    for j in range(om5.shape[1]):
        ome5[i,j] = more.decrypt(om5[i,j])
ome6 = np.zeros((om6.shape[0],om6.shape[1]))
for i in range(om6.shape[0]):
    for j in range(om6.shape[1]):
        ome6[i,j] = more.decrypt(om6[i,j])
        

# print(o1[0][0:10])
# print(ome1[0][0:10])
# print(o2[0][0:10])
# print(ome2[0][0:10])
# print(o3[0][0:10])
# print(ome3[0][0:10])
# print(o4[0][0:10])
# print(ome4[0][0:10])
# print(o5[0][0:10])
# print(ome5[0][0:10])
# print(o6[0][0:10])
# print(ome6[0][0:10])
print(om5[0][0:10])
print(om6[0][0:10])
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

# print("accuracy: ", accuracy/len(output_more))
# print("Correct: ", correct, ", incorrect: ", incorrect)