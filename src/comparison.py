import numpy as np

from network import Network
from activation_layer import ActivationLayer
from activation_functions import tanh
from loss_functions import mse
import csv
import time

import tenseal as ts

from progress.bar import Bar

autoencoder_plain = Network()
autoencoder_square = Network()
autoencoder_alt = Network()

autodecoder_plain = Network()
autodecoder_square = Network()
autodecoder_alt = Network()

autoencoder_plain.load("src/params/autoencoder_test")
autoencoder_square.load("src/params/autoencoder_ckks_square")
autoencoder_alt.load("src/params/autoencoder_ckks_alt")

for i in range(4):
    autodecoder_plain.add(autoencoder_plain.layers[i])
    autodecoder_square.add(autoencoder_square.layers[i])
    autodecoder_plain.add(autoencoder_plain.layers[i])

for i in range(4,9):
    autodecoder_plain.remove(-1)
    autodecoder_square.remove(-1)
    autodecoder_plain.remove(-1)

print(autoencoder_plain.layers)
print(autoencoder_square.layers)
print(autoencoder_alt.layers)

print(autodecoder_plain.layers)
print(autodecoder_square.layers)
print(autodecoder_alt.layers)


