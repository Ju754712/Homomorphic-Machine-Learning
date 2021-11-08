
import numpy as np

from progress.bar import Bar

from network import Network
from activation_layer import ActivationLayer
from activation_functions import sigmoid_more, sigmoid_prime_more, tanh_more, tanh_prime_more, square_more, square_prime_more
from schemes.more import MoreScheme

from keras.datasets import mnist
from keras.utils import np_utils

more = MoreScheme(2)
# Build Net with Square Function as activation function (Trained with MSE Error)
net_square = Network()
more_net_square = Network()
net_square.load("mnist_square")

# Build Net with Sigmoid Function as activation function (Trained with MSE Error)
net_sigmoid = Network()
more_net_sigmoid = Network()
net_sigmoid.load("mnist_sigmoid")

# Build Net with hyperbolic tangent as activation function (Trained with MSE Error)
net_tanh = Network()
more_net_tanh = Network()
net_tanh.load("mnist_tanh")

for i in range(3):
    # Build more Square Network
    more_net_square.add(net_square.layers[2*i]) 
    more_net_square.layers[2*i].encrypt_params_more(more)

    # Build More Sigmoid Network
    more_net_sigmoid.add(net_sigmoid.layers[2*i]) 
    more_net_sigmoid.layers[2*i].encrypt_params_more(more)
    
    #Build More Tanh Network
    more_net_tanh.add(net_tanh.layers[2*i]) 
    more_net_tanh.layers[2*i].encrypt_params_more(more)
    
    if i != 2:
        more_net_square.add(ActivationLayer(square_more, square_prime_more))
        more_net_sigmoid.add(ActivationLayer(sigmoid_more, sigmoid_prime_more))
        more_net_tanh.add(ActivationLayer(tanh_more, tanh_prime_more))
_, (x_test, y_test) = mnist.load_data()

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

x_test = x_test[0:100]
y_test = y_test[0:100]


input_more = np.zeros(x_test.shape, dtype=object)  
bar = Bar('Encrypting Test Data', max=x_test.shape[0])
for i in range(x_test.shape[0]):
    for k in range(x_test.shape[1]):                
        for m in range(x_test.shape[2]):
            input_more[i,k,m] = more.encrypt(x_test[i,k,m])
    bar.next()
bar.finish()

res_square = net_square.predict(x_test)
res_sigmoid = net_sigmoid.predict(x_test)
res_tanh = net_tanh.predict(x_test)

more_res_square = more_net_square.predict_more(input_more) 
more_res_sigmoid = more_net_sigmoid.predict_more(input_more) 
more_res_tanh = more_net_tanh.predict_more(input_more) 


dec_res_square = []
bar = Bar('Decrypting Square Test Results', max=len(more_res_square))
for i in range(len(more_res_square)):
    r = np.zeros(more_res_square[i].shape)
    for j in range(more_res_square[i].shape[0]):
        for k in range(more_res_square[i].shape[1]):
            r[j,k] = more.decrypt(more_res_square[i][j,k])
    dec_res_square.append(r)
    bar.next()
bar.finish()

dec_res_sigmoid = []
bar = Bar('Decrypting Sigmoid Test Results', max=len(more_res_sigmoid))
for i in range(len(more_res_sigmoid)):
    r = np.zeros(more_res_sigmoid[i].shape)
    for j in range(more_res_sigmoid[i].shape[0]):
        for k in range(more_res_sigmoid[i].shape[1]):
            r[j,k] = more.decrypt(more_res_sigmoid[i][j,k])
    dec_res_sigmoid.append(r)
    bar.next()
bar.finish()

dec_res_tanh = []
bar = Bar('Decrypting Tanh Test Results', max=len(more_res_tanh))
for i in range(len(more_res_tanh)):
    r = np.zeros(more_res_tanh[i].shape)
    for j in range(more_res_tanh[i].shape[0]):
        for k in range(more_res_tanh[i].shape[1]):
            r[j,k] = more.decrypt(more_res_tanh[i][j,k])
    dec_res_tanh.append(r)
    bar.next()
bar.finish()

acc_plain_square, acc_more_square, correct_plain_square, correct_more_square = 0,0,0,0
acc_plain_sigmoid, acc_more_sigmoid, correct_plain_sigmoid, correct_more_sigmoid = 0,0,0,0
acc_plain_tanh, acc_more_tanh, correct_plain_tanh, correct_more_tanh = 0,0,0,0

bar = Bar('Evaluating results', max=len(res_square))
for i in range(len(res_square)):
    acc_more_square += 1-abs(np.mean(y_test[i]-dec_res_square[i]))
    acc_plain_square += 1-abs(np.mean(y_test[i]-res_square[i]))

    acc_more_sigmoid += 1-abs(np.mean(y_test[i]-dec_res_sigmoid[i]))
    acc_plain_sigmoid += 1-abs(np.mean(y_test[i]-res_sigmoid[i]))

    acc_more_tanh += 1-abs(np.mean(y_test[i]-dec_res_tanh[i]))
    acc_plain_tanh += 1-abs(np.mean(y_test[i]-res_tanh[i]))

    true_value = np.argmax(y_test[i])

    more_value_square = np.argmax(dec_res_square[i])
    plain_value_square = np.argmax(res_square[i])

    more_value_sigmoid = np.argmax(dec_res_sigmoid[i])
    plain_value_sigmoid = np.argmax(res_sigmoid[i])

    more_value_tanh = np.argmax(dec_res_tanh[i])
    plain_value_tanh = np.argmax(res_tanh[i])

    if(true_value == plain_value_square):
        correct_plain_square += 1 
    if(true_value == more_value_square):
        correct_more_square += 1 

    if(true_value == plain_value_sigmoid):
        correct_plain_sigmoid += 1 
    if(true_value == more_value_sigmoid):
        correct_more_sigmoid += 1 

    if(true_value == plain_value_tanh):
        correct_plain_tanh += 1 
    if(true_value == more_value_tanh):
        correct_more_tanh += 1 

    bar.next()
bar.finish()


acc_more_square /= len(res_square)
acc_plain_square /= len(res_square)
acc_more_sigmoid /= len(res_sigmoid)
acc_plain_sigmoid /= len(res_sigmoid)
acc_more_sigmoid /= len(res_sigmoid)
acc_plain_sigmoid /= len(res_sigmoid)

print("Square Accuracy on Plaintext: ", acc_plain_square)
print("     where ", correct_plain_square, " of ", len(res_square), " data points where classified correctly")

print("Square Accuracy on Ciphertext: ", acc_more_square)
print("     where ", correct_more_square, " of ", len(res_square), " data points where classified correctly")

print("Sigmoid Accuracy on Plaintext: ", acc_plain_sigmoid)
print("     where ", correct_plain_sigmoid, " of ", len(res_sigmoid), " data points where classified correctly")

print("Sigmoid Accuracy on Ciphertext: ", acc_more_sigmoid)
print("     where ", correct_more_sigmoid, " of ", len(res_sigmoid), " data points where classified correctly")

print("Tanh Accuracy on Plaintext: ", acc_plain_tanh)
print("     where ", correct_plain_tanh, " of ", len(res_tanh), " data points where classified correctly")

print("Tanh Accuracy on Ciphertext: ", acc_more_tanh)
print("     where ", correct_more_tanh, " of ", len(res_tanh), " data points where classified correctly")