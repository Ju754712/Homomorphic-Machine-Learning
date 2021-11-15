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
data = np.random.randint(10,size=(1,10,1))
x_test = np.array([[[1],[2],[3],[4],[5],[6],[7],[8],[9]]])
print(data.shape)
print(x_test.shape)

arraylen = data.shape[1]  
model = keras.Sequential(
    [
        layers.Input(shape=(arraylen,1)),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2#, activation="relu"
        )
    ]
)

weights = model.get_weights()

net = Network()
net.add(Conv1DLayer(input_shape = data[0].shape, kernel=7, layer_depth = 32, strides = 2, padding ='same'))


net.layers[0].weights = weights[0]
net.layers[0].bias = weights[1]

for a in range(1):
    pred_keras = embedd_step(data[a, :, :].reshape((1,arraylen,1)), model) 
    pred_costum = net.predict(data[a, :, :].reshape((1,arraylen,1)))
    print(pred_keras[0][0])
    print(pred_costum[0][0])


