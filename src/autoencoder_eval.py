import numpy as np

import pickle

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer
from dropout_layer import DropoutLayer
from activation_functions import tanh, tanh_prime, relu, relu_prime, tanh_more, relu_more
from loss_functions import mse, mse_prime
from schemes.more import MoreScheme

from progress.bar import Bar

more = MoreScheme(2)

PATH = "./src/data/train.npy"
data = np.load(PATH, mmap_mode='r')
x_test = data[0:1]
print(x_test.shape)


net_plain = Network()
net_plain.load("./src/params/autoencoder")

net_more = Network()

# Add and encrypt first Conv Layer
net_more.add(net_plain.layers[0])
net_more.layers[-1].encrypt_params_more(more)
# Add activation layer
net_more.add(ActivationLayer(activation=relu_more, activation_prime=relu_prime))
# No Dropout Layer
# Add and encrypt second Conv Layer
net_more.add(net_plain.layers[3])
net_more.layers[-1].encrypt_params_more(more)
# Add Activation layer
net_more.add(ActivationLayer(activation=relu_more, activation_prime=relu_prime))

#Remove Decoder Levels
for i in reversed(range(5,10)):
    net_plain.remove(i)
#Remove Dropout Layer
net_plain.remove(2)

input_more = np.zeros((x_test.shape[0],x_test.shape[1], x_test.shape[2],2,2), dtype=np.float64)  
bar = Bar('Encrypting Test Data', max=x_test.shape[0])
for i in range(x_test.shape[0]):
    for k in range(x_test.shape[1]):                
        for m in range(x_test.shape[2]):
            input_more[i,k,m] = more.encrypt(x_test[i,k,m])
    bar.next()
bar.finish()

plain_output = net_plain.predict(x_test)
more_output = net_more.predict_more(input_more)

more_output_dec = []
for i in range(len(more_output)):
    enc = np.zeros((more_output[i].shape[0], more_output[i].shape[1]))
    for j in range(more_output[i].shape[0]):
        for k in range(more_output[i].shape[1]):
            enc[j,k] = more.decrypt(more_output[i][j,k])
    more_output_dec.append(enc)

for i in range(len(more_output_dec)):
    # Compute MSE
    more_output_dec[i] = np.nan_to_num(more_output_dec[i])
    print(mse(more_output_dec[i]-plain_output[i]))






## AUTODECODER EVALUATION



net_plain = Network()
net_plain.load("./src/params/autoencoder")

net_more = Network()

# Add and encrypt first Transposed Conv Layer
net_more.add(net_plain.layers[5])
net_more.layers[-1].encrypt_params_more(more)
# Add Activation Layer
net_more.add(ActivationLayer(activation=relu_more, activation_prime=relu_prime))
# No Dropout Layer
# Add and encrypt second Transposed Conv Layer
net_more.add(net_plain.layers[8])
net_more.layers[-1].encrypt_params_more(more)
# Add Tanh Activation Layer
net_more.add(ActivationLayer(activation=tanh_more, activation_prime=tanh_prime))

#Remove Encoder Levels
for i in range(0,5):
    net_plain.remove(0)
# Remove Dropout
net_plain.remove(2)

x_test_dec = np.array(plain_output)


input_more_dec = np.zeros((x_test_dec.shape[0],x_test_dec.shape[1], x_test_dec.shape[2],2,2), dtype=np.float64)  
bar = Bar('Encrypting Test Data', max=x_test_dec.shape[0])
for i in range(x_test_dec.shape[0]):
    for k in range(x_test_dec.shape[1]):                
        for m in range(x_test_dec.shape[2]):
            input_more_dec[i,k,m] = more.encrypt(x_test_dec[i,k,m])
    bar.next()
bar.finish()

plain_output = net_plain.predict(x_test_dec)
more_output = net_more.predict_more(input_more_dec)

more_output_dec = []
for i in range(len(more_output)):
    enc = np.zeros((more_output[i].shape[0], more_output[i].shape[1]))
    for j in range(more_output[i].shape[0]):
        for k in range(more_output[i].shape[1]):
            enc[j,k] = more.decrypt(more_output[i][j,k])
    more_output_dec.append(enc)

for i in range(len(more_output)):
    np.nan_to_num(more_output_dec[i])
    print(np.average(more_output_dec[i]-plain_output[i]))
    print(np.max(more_output_dec[i]-plain_output[i]))
    print(more_output_dec[i][0:10][0:10])