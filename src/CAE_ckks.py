import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pickle
import tensorflow as tf

import tenseal as ts


from network import Network
from activation_layer import ActivationLayer
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer
from dropout_layer import DropoutLayer
from activation_functions import sigmoid_approx, square, square_prime, tanh, tanh_prime

from keras.models import Sequential
from keras.layers import Dense

# Custom activation function
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


def square_activation(x):
    return x*x

def sigmoid_approx_activation(x):
    return -0.004 * x**3 + 0.197*x +0.5




def embedd_step(data_test, model, clear=True):
    embedding = model.predict(data_test)
    # reconstruction_err = err_function(data_test, embedding).numpy()

    return embedding

def get_model(inputshape, lr=0.0001):
    model = keras.Sequential(
        [
            layers.Input(shape=(arraylen,1)),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation='square_activation'
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="square_activation"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="square_activation"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="sigmoid_approx_activation"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"
            )
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")

    return model

def get_trained_model():
    model = keras.Sequential(
    [
        layers.Input(shape=(arraylen,1)),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation='square_activation'
        ),
        
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="square_activation"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="square_activation"
        ),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="sigmoid_approx_activation"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"
        )
    ])
    return model

if __name__ == "__main__":
    get_custom_objects().update({'square_activation': Activation(square_activation)})
    get_custom_objects().update({'sigmoid_approx_activation': Activation(sigmoid_approx_activation)})
    PATH = "./src/data/train.npy"
    ERROR_SAVE_NAME = "../CAE/sr"
    EXP = 'Finova2'
    EPOCHS2TRAIN = 50
    BATCHSIZE = 4
    ERR_FNCT = tf.keras.losses.MeanSquaredError()
    FEATURE = "combined"
    TRAINON = 'all' # or 'all'
    MODE = 'splits'# 'splits' or 'trainon'
    SAVE = True

    data = np.load(PATH, mmap_mode='r') # load data
    print(data.shape)
    if TRAINON == 'all':
        TRAINON = data.shape[0] 
    arraylen = 120 
    model = get_model(arraylen)
    model.summary() # generate autoencoder model
    history = model.fit( # train autoencoder 
    data[0:TRAINON, 0:arraylen, :],
    data[0:TRAINON, 0:arraylen, :],
    epochs=EPOCHS2TRAIN,
    batch_size=BATCHSIZE,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )     


    weights = model.get_weights()

    net = Network()
    net.add(Conv1DLayer(input_shape = (arraylen,1), kernel=7, layer_depth = 32, strides = 2, padding ='same'))
    net.add(ActivationLayer(activation = square, activation_prime = square_prime))
    net.add(Conv1DLayer(input_shape = (arraylen/2,32), kernel = 7, layer_depth = 16, strides = 2, padding = 'same'))
    net.add(ActivationLayer(activation = square, activation_prime = square))
    net.add(Conv1DTransposedLayer(input_shape = (arraylen/4,16) , kernel = 7, layer_depth = 16, strides=2, padding = 'same', a=1))
    net.add(ActivationLayer(activation=square, activation_prime=square_prime))
    net.add(Conv1DTransposedLayer(input_shape=(arraylen/2,16), kernel = 7, layer_depth=32, strides=2, padding='same', a=1))
    net.add(ActivationLayer(activation=sigmoid_approx, activation_prime=sigmoid_approx))
    net.add(Conv1DTransposedLayer(input_shape=(arraylen,32), kernel=7, layer_depth=1,  strides=1, padding='same', a=0))

    trained_model = get_trained_model()
    trained_model.layers[0].set_weights([weights[0], weights[1]])
    trained_model.layers[1].set_weights([weights[2], weights[3]])
    trained_model.layers[2].set_weights([weights[4], weights[5]])
    trained_model.layers[3].set_weights([weights[6], weights[7]])
    trained_model.layers[4].set_weights([weights[8], weights[9]])

    net.layers[0].weights = weights[0]
    net.layers[0].bias = weights[1]
    net.layers[2].weights = weights[2]
    net.layers[2].bias = weights[3]
    net.layers[4].weights = np.flip(weights[4].transpose(0,2,1),0)
    net.layers[4].bias = weights[5]
    net.layers[6].weights = np.flip(weights[6].transpose(0,2,1),0)
    net.layers[6].bias = weights[7]
    net.layers[8].weights = np.flip(weights[8].transpose(0,2,1),0)
    net.layers[8].bias = weights[9]

    if SAVE == True:
        model.save('./src/keras_model/Autoencoder_ckks_square')
        net.save("./src/params/autoencoder_ckks_square")





    ## Encryption Parameters

    # controls precision of the fractional part


