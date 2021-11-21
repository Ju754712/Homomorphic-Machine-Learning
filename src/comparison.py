import numpy as np

from network import Network
from activation_layer import ActivationLayer
from activation_functions import tanh
from loss_functions import mse
import csv
import time

import tenseal as ts

from progress.bar import Bar

PATH = "./src/data/train.npy"

data = np.load(PATH, mmap_mode='r') 

autoencoder_plain = Network()
autoencoder_square = Network()
autoencoder_alt = Network()

autodecoder_plain = Network()
autodecoder_square = Network()
autodecoder_alt = Network()

autoencoder_plain.load("src/params/autoencoder_test")
autoencoder_square.load("src/params/autoencoder_ckks_square")
autoencoder_alt.load("src/params/autoencoder_ckks_alt")

for i in range(4,9):
    autodecoder_plain.add(autoencoder_plain.layers[i])
    autodecoder_square.add(autoencoder_square.layers[i])
    autodecoder_alt.add(autoencoder_alt.layers[i])

for i in range(4,9):
    autoencoder_plain.remove(-1)
    autoencoder_square.remove(-1)
    autoencoder_alt.remove(-1)

encoding_plain = autoencoder_plain.predict(data[0:1,0:120,:].reshape(1,120,1))
encoding_square = autoencoder_square.predict(data[0:1,0:120,:].reshape(1,120,1))
encoding_alt = autoencoder_alt.predict(data[0:1,0:120,:].reshape(1,120,1))

encoding_accuracy_plain = mse(data[0][0:120,:], encoding_plain[0])

print(data[0][0:120,:].shape)
print(encoding_plain[0].shape)
print(encoding_accuracy_plain)



