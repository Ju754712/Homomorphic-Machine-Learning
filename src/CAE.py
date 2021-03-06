import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pickle
import tensorflow as tf
import json

from network import Network
from activation_layer import ActivationLayer
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer
from dropout_layer import DropoutLayer
from activation_functions import tanh, tanh_prime, relu, relu_prime


def embedd_step(data_test, model, err_function, clear=True):
    embedding = model.predict(data_test)
    reconstruction_err = err_function(data_test, embedding).numpy()

    return reconstruction_err

def get_model(arraylen, lr=0.0001):
    model = keras.Sequential(
        [
            layers.Input(shape=(arraylen,1)),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=2, activation="tanh"
            ),
            layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"
            )
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")

    return model

def get_trained_model(arraylen):
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
        ),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="tanh"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"
        )
    ])
    return model

if __name__ == "__main__":

    PATH = "./src/data/finova.npy"
    ERROR_SAVE_NAME = "../CAE/sr"
    EXP = 'Finova2'
    EPOCHS2TRAIN = 50
    BATCHSIZE = 4
    ERR_FNCT = tf.keras.losses.MeanSquaredError()
    FEATURE = "combined"
    TRAINON = 10000 # or 'all'
    MODE = 'splits'# 'splits' or 'trainon'
    SAVE = True

    data = np.load(PATH, mmap_mode='r') # load data
    if TRAINON == 'all':
        TRAINON = data.shape[0] 
    arraylen = data.shape[1]   
    model = get_trained_model(arraylen)
    model.summary() # generate autoencoder model
    # history = model.fit( # train autoencoder 
    # data[0:TRAINON, :, :],
    # data[0:TRAINON, :, :],
    # epochs=EPOCHS2TRAIN,
    # batch_size=BATCHSIZE,
    # validation_split=0.1,
    # callbacks=[
    #     keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    #     ],
    # )     


    # weights = model.get_weights()

    net = Network()
    net.load('./src/params/autoencoder')
    model.layers[0].set_weights([net.layers[0].weights, net.layers[0].bias])
    model.layers[1].set_weights([net.layers[2].weights, net.layers[2].bias])
    model.layers[2].set_weights([np.flip(net.layers[4].weights.transpose(0,2,1),0), net.layers[4].bias])
    model.layers[3].set_weights([np.flip(net.layers[6].weights.transpose(0,2,1),0), net.layers[6].bias])
    model.layers[4].set_weights([np.flip(net.layers[8].weights.transpose(0,2,1),0), net.layers[8].bias])
    # net.add(Conv1DLayer(input_shape = data[0].shape, kernel=7, layer_depth = 32, strides = 2, padding ='same'))
    # net.add(ActivationLayer(activation = relu, activation_prime = relu_prime))
    # net.add(Conv1DLayer(input_shape = (75000,32), kernel = 7, layer_depth = 16, strides = 2, padding = 'same'))
    # net.add(ActivationLayer(activation = relu, activation_prime = relu_prime))
    # net.add(Conv1DTransposedLayer(input_shape = (37500,16) , kernel = 7, layer_depth = 16, strides=2, padding = 'same', a=1))
    # net.add(ActivationLayer(activation=relu, activation_prime=relu_prime))
    # net.add(Conv1DTransposedLayer(input_shape=(75000,16), kernel = 7, layer_depth=32, strides=2, padding='same', a=1))
    # net.add(ActivationLayer(activation=tanh, activation_prime=tanh_prime))
    # net.add(Conv1DTransposedLayer(input_shape=(150000,32), kernel=7, layer_depth=1,  strides=1, padding='same', a=0))

    # trained_model = get_trained_model()
    # trained_model.layers[0].set_weights([weights[0], weights[1]])
    # trained_model.layers[1].set_weights([weights[2], weights[3]])
    # trained_model.layers[2].set_weights([weights[4], weights[5]])
    # trained_model.layers[3].set_weights([weights[6], weights[7]])
    # trained_model.layers[4].set_weights([weights[8], weights[9]])

    # net.layers[0].weights = weights[0]
    # net.layers[0].bias = weights[1]
    # net.layers[2].weights = weights[2]
    # net.layers[2].bias = weights[3]
    # net.layers[4].weights = np.flip(weights[4].transpose(0,2,1),0)
    # net.layers[4].bias = weights[5]
    # net.layers[6].weights = np.flip(weights[6].transpose(0,2,1),0)
    # net.layers[6].bias = weights[7]
    # net.layers[8].weights = np.flip(weights[8].transpose(0,2,1),0)
    # net.layers[8].bias = weights[9]

    # if SAVE == True:
    #     model.save('./src/keras_model/Autoencoder')
    #     print('Saved')
    # model = keras.models.load_model('./src/keras_model/Autoencoder')

    # err_data = [] # error array
    # pred = [] # predictions array  
    pred = []
    err_data = []   #Pickling
    for a in range(1600):
        print(a)            
        tmp = embedd_step(data[10*a, :, :].reshape((1,arraylen,1)), model, ERR_FNCT) 
        # pred.append(tmp[0])
        err_data.append(tmp)
    with open("./src/err_data", "wb") as fp:
        pickle.dump(err_data, fp)

    plt.plot(err_data)
    plt.xlabel("Sample")
    plt.ylabel("Reconstruction Error [MSE]")
    plt.savefig(f'mse_small.png')
    plt.close()
    plt.plot(np.cumsum(err_data))
    plt.xlabel("Sample")
    plt.ylabel("Cumulative Reconstruction Error [MSE]")
    plt.savefig(f'mse_small_cumsum.png')
    # plt.close()
    # pickle.dump(err_data, open(f"{ERROR_SAVE_NAME}{EXP}{FEATURE}{TRAINON}_err", "wb"))
    # pickle.dump(pred, open(f"{ERROR_SAVE_NAME}{EXP}{FEATURE}{TRAINON}_pred", "wb"))



