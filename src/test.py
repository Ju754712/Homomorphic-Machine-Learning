import numpy as np
from tensorflow import keras

import tensorflow as tf
from network import Network 
from tensorflow.keras import layers

DATA_PATH = "./src/data/train.npy"
ERR_FNCT = tf.keras.losses.MeanSquaredError()
data = np.load(DATA_PATH, mmap_mode='r') # load data
model = keras.models.load_model('./src/keras_model/Autoencoder')

arraylen = data.shape[1] 

err_data = [] # error array
pred_keras = [] # predictions array  

def embedd_step(data_test, model, clear=True):
    embedding = model.predict(data_test)

    return embedding

weights = model.get_weights()
model_test = keras.Sequential(
    [
        layers.Input(shape=(arraylen,1)),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        )
        # layers.Dropout(rate=0.2),
        # layers.Conv1D(
        #     filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        # ),
        # layers.Conv1DTranspose(
        #     filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        # ),
        # layers.Dropout(rate=0.2),
        # layers.Conv1DTranspose(
        #     filters=32, kernel_size=7, padding="same", strides=2, activation="tanh"
        # ),
        # layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"
        # ),
        # #layers.Cropping1D(cropping=1
        # #)
    ]
)


model_test.layers[0].set_weights([weights[0], weights[1]])

model_test.summary()


tmp = embedd_step(data[0, :, :].reshape((1,arraylen,1)), model_test) 
pred_keras.append(tmp)

net = Network()
net.load('./src/params/autoencoder')
for i in range(9):
    net.remove(-1)
print(len(net.layers))

pred_costum = net.predict(data[0:1])

print(pred_keras[0].shape)
print(pred_costum[0].shape)
for i in range(pred_costum[0][0].shape[0]):
    print(pred_keras[0][0][1][i])
    print(pred_costum[0][0][i])

