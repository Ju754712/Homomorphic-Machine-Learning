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
from fc_layer import FCLayer
from activation_functions import sigmoid_approx, sigmoid_approx_prime, square

from keras.models import Sequential
from keras.layers import Dense

# Custom activation function
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.datasets import mnist
from keras.utils import np_utils


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
            layers.Input(shape=(28,28,1)),                    
            layers.Conv2D(filters=4, kernel_size=7, padding='valid', strides=3, activation="square_activation"),
            layers.Reshape((256,)),
            layers.Dense(64, activation = 'square_activation'),
            layers.Dense(10),
        ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")

    return model

def get_trained_model(inputshape):
    model = keras.Sequential(
        [
            layers.Input(shape=(28,28,1)),                    
            layers.Conv2D(filters=4, kernel_size=7, padding='valid', strides=3, activation="square_activation"),
            layers.Reshape((256,)),
            layers.Dense(64, activation = 'square_activation'),
            layers.Dense(10),
        ])
    return model

if __name__ == "__main__":
    get_custom_objects().update({'square_activation': Activation(square_activation)})
    get_custom_objects().update({'sigmoid_approx_activation': Activation(sigmoid_approx_activation)})
    
    EPOCHS = 35
    BATCH_SIZE = 1
    LEARNING_RATE = 0.1
    (x_train, y_train), (x_test, y_test) = mnist.load_data()


    # training data : 60000 samples
    # # reshape and normalize input data
    print(x_train.shape)
    x_train = x_train.reshape(x_train.shape[0],28,28,1)
    x_train = x_train.astype('float32')
    print(x_train.shape)
    # x_train /= 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train)

    # same for test data : 10000 samples
    # x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    # x_test /= 255
    y_test = np_utils.to_categorical(y_test)


    # x_train = x_train[0:4000]
    # y_train = y_train[0:1000]

    # x_train = x_train[0:1000]
    # y_train = y_train[0:1000]

    model = get_model(x_train.shape)
    model.summary() # generate autoencoder model
    history = model.fit( # train autoencoder 
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        # callbacks=[
        #     keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        # ],
    )     


    weights = model.get_weights()
    for i in range(6):
        print(weights[i].shape)
    # net = Network()
    # net.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
    # net.add(ActivationLayer(square, sigmoid_approx_prime))
    # net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
    # net.add(ActivationLayer(square, sigmoid_approx_prime))
    # net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
    # net.add(ActivationLayer(square, sigmoid_approx_prime))

    # net.layers[0].weights=weights[0]
    # net.layers[0].bias[0]=weights[1]
    # net.layers[2].weights=weights[2]
    # net.layers[2].bias[0]=weights[3]
    # net.layers[4].weights=weights[4]
    # net.layers[4].bias[0]=weights[5]

    # net.save("src/params/mnist_tenseal")
    pickle.dump(weights, open("mnist_tenseal.p", "wb" ) )
