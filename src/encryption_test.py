import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer
from dropout_layer import DropoutLayer
from activation_functions import tanh, tanh_prime, relu, relu_prime, tanh_more, relu_more
from loss_functions import mse, mse_prime
from schemes.more import MoreScheme
import csv
import time

from progress.bar import Bar

more = MoreScheme(2)

PATH = "./src/data/train.npy"
data = np.load(PATH, mmap_mode='r')
x_test = data[0:100]

print(x_test.shape)
time1 = time.time()
enc = more.encrypt_array(x_test)
time2 = time.time()

print(time2-time1)