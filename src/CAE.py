import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pickle
import tensorflow as tf


from network import Network
from activation_layer import ActivationLayer
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer
from dropout_layer import DropoutLayer
from activation_functions import tanh, tanh_prime, relu, relu_prime


def embedd_step(data_test, model, err_function, clear=True):
    embedding = model.predict(data_test)
    reconstruction_err = err_function(data_test, embedding).numpy()

    return embedding, reconstruction_err

def get_model(inputshape, lr=0.0001):
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
            ),
            #layers.Cropping1D(cropping=1
            #)
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse")

    return model

if __name__ == "__main__":

    PATH = "./src/data/train.npy"
    ERROR_SAVE_NAME = "../CAE/sr"
    EXP = 'Finova2'
    EPOCHS2TRAIN = 2
    BATCHSIZE = 4
    ERR_FNCT = tf.keras.losses.MeanSquaredError()
    FEATURE = "combined"
    TRAINON = 10 # or 'all'
    MODE = 'splits'# 'splits' or 'trainon'
    SAVE = False

    data = np.load(PATH, mmap_mode='r') # load data
    print(data.shape)
    if TRAINON == 'all':
        TRAINON = data.shape[0] 
    arraylen = data.shape[1]   
    model = get_model(arraylen)
    model.summary() # generate autoencoder model
    history = model.fit( # train autoencoder 
    data[0:TRAINON, :, :],
    data[0:TRAINON, :, :],
    epochs=EPOCHS2TRAIN,
    batch_size=BATCHSIZE,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
        ],
    )      
    if SAVE == True:
        model.save('/hpcwork/mu637455/Code/Autoencoder/')
    weights = model.get_weights()

    net = Network()
    net.add(Conv1DLayer(input_shape = data[0].shape, kernel=7, layer_depth = 32, strides = 2, padding ='same'))
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

    print(net.layers[0].weights.shape, net.layers[0].bias.shape)
    net.layers[0].weights = weights[0]
    net.layers[0].bias = weights[1]
    print(net.layers[0].weights.shape, net.layers[0].bias.shape)
    print(net.layers[3].weights.shape, net.layers[3].bias.shape)
    net.layers[3].weights = weights[2]
    net.layers[3].bias = weights[3]
    print(net.layers[3].weights.shape, net.layers[3].bias.shape)
    print(net.layers[5].weights.shape, net.layers[5].bias.shape)
    net.layers[5].weights = weights[4].transpose(0,2,1)
    net.layers[5].bias = weights[5]
    print(net.layers[5].weights.shape, net.layers[5].bias.shape)
    print(net.layers[8].weights.shape, net.layers[8].bias.shape)
    net.layers[8].weights = weights[6].transpose(0,2,1)
    net.layers[8].bias = weights[7]
    print(net.layers[8].weights.shape, net.layers[8].bias.shape)
    print(net.layers[10].weights.shape, net.layers[10].bias.shape)
    net.layers[10].weights = weights[8].transpose(0,2,1)
    net.layers[10].bias = weights[9]
    print(net.layers[10].weights.shape, net.layers[10].bias.shape)
    net.save("./src/params/autoencoder")
    # model = keras.models.load_model('/hpcwork/mu637455/Code/Autoencoder/')

    # err_data = [] # error array
    # pred = [] # predictions array  

    # for a in range(data.shape[0]-TRAINON):
    #         tmp = embedd_step(data[TRAINON+a, :, :].reshape((1,arraylen,1)), model, ERR_FNCT) 
    #         err_data.append(tmp[1])
    #         pred.append(tmp[0])

    # plt.plot(err_data)
    # plt.savefig(f'{ERROR_SAVE_NAME}{EXP}{FEATURE}{TRAINON}.png')
    # plt.close()
    # plt.plot(np.cumsum(err_data))
    # plt.savefig(f'{ERROR_SAVE_NAME}{EXP}{FEATURE}{TRAINON}_cumsum.png')
    # plt.close()
    # pickle.dump(err_data, open(f"{ERROR_SAVE_NAME}{EXP}{FEATURE}{TRAINON}_err", "wb"))
    # pickle.dump(pred, open(f"{ERROR_SAVE_NAME}{EXP}{FEATURE}{TRAINON}_pred", "wb"))



