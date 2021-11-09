import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer
from dropout_layer import DropoutLayer
from activation_functions import tanh, tanh_prime, relu, relu_prime
from loss_functions import mse, mse_prime


#Configuration
PATH = "./src/data/train.npy"
EPOCHS2TRAIN = 50
BATCHSIZE = 4
TRAINON = 999 # or 'all'

#Load Input 
data = np.load(PATH, mmap_mode='r')
print(data.shape)
train_data = data
test_data = data[TRAINON:TRAINON+10]


#network
net = Network()
net.add(Conv1DLayer(input_shape = train_data[0].shape, kernel=7, layer_depth = 32, strides = 2, padding ='same'))
net.add(ActivationLayer(activation = relu, activation_prime = relu_prime))
net.add(DropoutLayer(rate = 0.2))
net.add(Conv1DLayer(input_shape = (75000,32), kernel = 7, layer_depth = 16, strides = 2, padding = 'same'))
net.add(ActivationLayer(activation = relu, activation_prime = relu_prime))
net.add(Conv1DTransposedLayer(input_shape = (37500,16) , kernel = 7, layer_depth = 16, strides=2, padding = 'same', a=1))
net.add(ActivationLayer(activation=relu, activation_prime=relu_prime))
net.add(DropoutLayer(rate=0.2))
net.add(Conv1DTransposedLayer(input_shape=(75000,16), kernel = 7, layer_depth=32, strides=2, padding='same', a=1))
net.add(ActivationLayer(activation=tanh, activation_prime=tanh_prime))
net.add(Conv1DTransposedLayer(input_shape=(150000,32), kernel=7, layer_depth=1,  strides=1, padding='same', a=0))

#train
net.use(loss=mse, loss_prime=mse_prime)
net.fit(train_data, train_data, epochs=EPOCHS2TRAIN, batch_size = BATCHSIZE, learning_rate=0.2)


net.save("src/params/autoencoder")


