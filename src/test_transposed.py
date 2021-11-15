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
data = np.array([[[1],[2],[3],[4],[5],[6]]])


arraylen = data.shape[1]  
#change kernel back
model = keras.Sequential(
    [
        layers.Input(shape=(arraylen,1)),
        layers.Conv1DTranspose(
            filters=1, kernel_size=5, padding="same", strides=2
        )        

    ]
)


weight = np.array([[[1]],[[2]],[[3]],[[4]],[[5]]])
bias = np.array([0])
model.layers[0].set_weights([weight,bias])
weights = model.get_weights()
#change kernel back
net = Network()
net.add(Conv1DTransposedLayer(input_shape = (6,1) , kernel = 5, layer_depth = 1, strides=2, padding = 'same', a=1))



net.layers[0].weights = np.flip(np.transpose(weights[0],(0,2,1)),0)
net.layers[0].bias = weights[1]

for a in range(1):
    pred_keras = embedd_step(data[a, :, :].reshape((1,arraylen,1)), model) 
    pred_costum = net.predict(data[a, :, :].reshape((1,arraylen,1)))
    print(pred_keras[0])
    print(pred_costum[0])