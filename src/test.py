import numpy as np
from tensorflow import keras

import tensorflow as tf
from activation_layer import ActivationLayer
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer
from activation_functions import tanh, tanh_prime, relu, relu_prime
from network import Network 
from tensorflow.keras import layers
from data_setup import random_data

def embedd_step(data_test, model, clear=True):
    embedding = model.predict(data_test)

    return embedding
DATA_PATH = "./src/data/train.npy"
ERR_FNCT = tf.keras.losses.MeanSquaredError()
data = np.load(DATA_PATH, mmap_mode='r') # load data
# data = np.random.randint(10,size=(1,10,1))
# data = np.array([[[1],[2],[3],[4],[5],[6]]])


arraylen = data.shape[1]  
#change kernel back
model = keras.Sequential(
    [
        layers.Input(shape=(arraylen,1)),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        )        
        # layers.Conv1DTranspose(
        #     filters=32, kernel_size=7, padding="same", strides=2, activation="tanh"
        # ),
        # layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"
        # ),
    ]
)

weights = model.get_weights()

#change kernel back
net = Network()
net.add(Conv1DLayer(input_shape = data[0].shape, kernel=7, layer_depth = 32, strides = 2, padding ='same'))
net.add(ActivationLayer(activation = relu, activation_prime = relu_prime)) 
net.add(Conv1DLayer(input_shape = (75000,32), kernel = 7, layer_depth = 16, strides = 2, padding = 'same'))
net.add(ActivationLayer(activation = relu, activation_prime = relu_prime))
net.add(Conv1DTransposedLayer(input_shape = (37500,16) , kernel = 7, layer_depth = 16, strides=2, padding = 'same', a=1))
net.add(ActivationLayer(activation=relu, activation_prime=relu_prime))
# net.add(Conv1DTransposedLayer(input_shape=(75000,16), kernel = 7, layer_depth=32, strides=2, padding='same', a=1))
# net.add(ActivationLayer(activation=tanh, activation_prime=tanh_prime))
# net.add(Conv1DTransposedLayer(input_shape=(150000,32), kernel=7, layer_depth=1,  strides=1, padding='same', a=0))



net.layers[0].weights = weights[0]
net.layers[0].bias = weights[1]
net.layers[2].weights = weights[2]
net.layers[2].bias = weights[3]
net.layers[4].weights = np.flip(np.transpose(weights[4],(0,2,1)),0)
net.layers[4].bias = weights[5]

for a in range(1):
    pred_keras = embedd_step(data[a, :, :].reshape((1,arraylen,1)), model) 
    pred_costum = net.predict(data[a, :, :].reshape((1,arraylen,1)))
    print(pred_keras[0][0])
    print(pred_costum[0][0])

# Check whole net pls


