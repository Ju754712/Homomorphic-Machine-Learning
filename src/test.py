import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pickle
import tensorflow as tf
import time

import tenseal as ts

from loss_functions import mse
from network import Network
from activation_layer import ActivationLayer
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer
from dropout_layer import DropoutLayer
from activation_functions import square, square_ckks, square_prime, tanh, tanh_prime, sigmoid_approx, sigmoid_approx_ckks

from keras.models import Sequential
from keras.layers import Dense

# Custom activation function
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

PATH = "./src/data/train.npy"


data = np.load(PATH, mmap_mode='r') # load data

data = data[0:1,0:20,:]

print(data.shape)

array_length = data.shape[1]

autoencoder = Network()
autoencoder.add(Conv1DLayer(input_shape = (array_length,1), kernel=7, layer_depth = 32, strides = 2, padding ='same'))
autoencoder.add(ActivationLayer(activation = square, activation_prime = square_prime))
autoencoder.add(Conv1DLayer(input_shape = (array_length/2,32), kernel = 7, layer_depth = 16, strides = 2, padding = 'same'))
autoencoder.add(ActivationLayer(activation = square, activation_prime = square))
autodecoder = Network()
autodecoder.add(Conv1DTransposedLayer(input_shape = (array_length/4,16) , kernel = 7, layer_depth = 16, strides=2, padding = 'same', a=1))
autodecoder.add(ActivationLayer(activation=square, activation_prime=square_prime))
autodecoder.add(Conv1DTransposedLayer(input_shape=(array_length/2,16), kernel = 7, layer_depth=32, strides=2, padding='same', a=1))
autodecoder.add(ActivationLayer(activation=sigmoid_approx, activation_prime=sigmoid_approx))
autodecoder.add(Conv1DTransposedLayer(input_shape=(array_length,32), kernel=7, layer_depth=1,  strides=1, padding='same', a=0))

autoencoder_ckks = Network()
autoencoder_ckks.add(autoencoder.layers[0])
autoencoder_ckks.add(ActivationLayer(activation=square_ckks, activation_prime=square_ckks))
autoencoder_ckks.add(autoencoder.layers[2])
autoencoder_ckks.add(ActivationLayer(activation=square_ckks, activation_prime=square_ckks))
autodecoder_ckks = Network()
autodecoder_ckks.add(autodecoder.layers[0])
autodecoder_ckks.add(ActivationLayer(activation=square_ckks, activation_prime=square_ckks))
autodecoder_ckks.add(autodecoder.layers[2])
autodecoder_ckks.add(ActivationLayer(activation=sigmoid_approx_ckks,activation_prime=sigmoid_approx_ckks))
autodecoder_ckks.add(autodecoder.layers[4])


# Create TenSEAL context
bits_scale = 26
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)
#set the scale
context.global_scale = pow(2, bits_scale)


for i in range(data.shape[0]):
    print("Encryption")
    x_test_ckks = np.zeros((1,data.shape[1], data.shape[2]), dtype=object)
    time1 = time.time()
    for k in range(data[i].shape[0]):
        for j in range(data[i].shape[1]):
            x_test_ckks[0,k,j] = ts.CKKSVector(context, [data[i,k,j]])
    time2 = time.time()
    encoder_input_encryption_time = time2-time1
    print("Encoder Input Encryption Time: ", encoder_input_encryption_time)

    time1 = time.time()
    encoding_plain = autoencoder.predict(data[i,:,:].reshape((1,array_length,1)))
    time2 = time.time()
    encoding_ckks_enc = autoencoder_ckks.predict_ckks(x_test_ckks)
    time3 = time.time()

    print("Plain: ", time2-time1)
    print("Encrypted: ", time3-time2)

    print(encoding_ckks_enc[0].shape)

    time1 = time.time()
    encoding_ckks = np.zeros((1,encoding_ckks_enc[0].shape[0], encoding_ckks_enc[0].shape[1]))
    for k in range(encoding_ckks_enc[0].shape[0]):
        for j in range(encoding_ckks_enc[0].shape[1]):
            encoding_ckks[0,k,j] = encoding_ckks_enc[0][k,j].decrypt()[0]

    time2 = time.time()

    encoder_output_decryption_time = time2-time1
    encoding_accuracy = mse(encoding_plain[0], encoding_ckks[0])
    print("Encoder Output Decryption Time: ", encoder_output_decryption_time)
    print(encoding_accuracy)

    time1 = time.time()
    encoding_ckks_enc = np.zeros((encoding_ckks.shape[0],encoding_ckks.shape[1], encoding_ckks.shape[2]),dtype=object)
    for k in range(encoding_ckks[0].shape[0]):
        for j in range(encoding_ckks[0].shape[1]):
            encoding_ckks_enc[0,k,j] = ts.CKKSVector(context, [encoding_ckks[0,k,j]])
    time2 = time.time()

    decoder_input_encryption_time = time2-time1

    time1 = time.time()
    decoding_plain = autodecoder.predict(encoding_plain)
    time2 = time.time()
    decoding_ckks_enc = autodecoder_ckks.predict_ckks(encoding_ckks_enc)
    time3 = time.time()

    decoder_plain_time = time2-time1
    decoder_ckks_time = time3-time2

    print("Decoder Plain took", decoder_plain_time, " secs")
    print("Decoder CKKS took", decoder_ckks_time, " secs")
        
    time1 = time.time()
    decoding_ckks = np.zeros((1,decoding_ckks_enc[0].shape[0], decoding_ckks_enc[0].shape[1]))
    for k in range(decoding_ckks_enc[0].shape[0]):
        for j in range(decoding_ckks_enc[0].shape[1]):
            decoding_ckks[0,k,j] = decoding_ckks_enc[0][k,j].decrypt()[0]

    time2 = time.time()
    decoder_output_decryption_time = time2-time1
    print("Decryption took ", decoder_output_decryption_time, " secs")

    print(decoding_plain[0].shape, decoding_ckks[0].shape)
    
    decoding_accuracy = mse(decoding_plain[0], decoding_ckks[0])

    print("Accuracies: ", decoding_accuracy)