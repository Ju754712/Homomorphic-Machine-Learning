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


autoencoder_plain = Network()
autoencoder_plain.load("./src/params/autoencoder")

autoencoder_more = Network()

# Add and encrypt first Conv Layer
autoencoder_more.add(autoencoder_plain.layers[0])
# Add activation layer
autoencoder_more.add(ActivationLayer(activation=relu_more, activation_prime=relu_prime))
# No Dropout Layer
# Add and encrypt second Conv Layer
autoencoder_more.add(autoencoder_plain.layers[2])
# Add Activation layer
autoencoder_more.add(ActivationLayer(activation=relu_more, activation_prime=relu_prime))

for i in reversed(range(4,9)):
    autoencoder_plain.remove(i)

autodecoder_plain = Network()
autodecoder_plain.load("./src/params/autoencoder")

autodecoder_more = Network()


# Add and encrypt first Transposed Conv Layer
autodecoder_more.add(autodecoder_plain.layers[4])
# Add Activation Layer
autodecoder_more.add(ActivationLayer(activation=relu_more, activation_prime=relu_prime))
# No Dropout Layer
# Add and encrypt second Transposed Conv Layer
autodecoder_more.add(autodecoder_plain.layers[6])
# Add Tanh Activation Layer
autodecoder_more.add(ActivationLayer(activation=tanh_more, activation_prime=tanh_prime))
autodecoder_more.add(autodecoder_plain.layers[8])

for i in reversed(range(0,4)):
    autodecoder_plain.remove(0)

with open('./src/csv/autoencoder_more.csv', 'w', newline='') as csvfile:
    fieldnames = ['encoding_accuracy', 'decoding_accuracy_plain', 'decoding_accuracy_more', 'decoding_accuracy', 'encoder_input_encryption_time', 'encoder_plain_time', 'encoder_more_time', 'encoder_output_decryption_time', 'decoder_input_encryption_time', 'decoder_plain_time', 'decoder_more_time', 'decoder_output_decryption_time' ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    bar = Bar("Processing...", max = x_test.shape[0])
    for i in range(x_test.shape[0]):
        bar.next()
        #Encryption
        x_test_more = np.zeros((1,x_test.shape[1], x_test.shape[2],2,2))
        time1 = time.time()
        for k in range(x_test[i].shape[0]):
            for j in range(x_test[i].shape[1]):
                x_test_more[0,k,j] = more.encrypt(x_test[i,k,j])
        time2 = time.time()

        encoder_input_encryption_time = time2-time1

        time1 = time.time()
        encoding_plain = autoencoder_plain.predict(x_test)
        time2 = time.time()
        encoding_more_enc = autoencoder_more.predict_more(x_test_more)
        time3 = time.time()

        encoder_plain_time = time2-time1
        encoder_more_time = time3-time2

        time1 = time.time()
        encoding_more = np.zeros((1,encoding_more_enc[0].shape[0], encoding_more_enc[0].shape[1]))
        for k in range(encoding_more_enc[0].shape[0]):
            for j in range(encoding_more_enc[0].shape[1]):
                encoding_more[0,k,j] = more.decrypt(encoding_more_enc[0][k,j])

        time2 = time.time()

        encoding_more[0] = np.nan_to_num(encoding_more[0])
        encoder_output_decryption_time = time2-time1
        encoding_accuracy = mse(encoding_plain[0], encoding_more[0])
        print(encoding_accuracy)

        time1 = time.time()
        encoding_more_enc = np.zeros((encoding_more.shape[0],encoding_more.shape[1], encoding_more.shape[2],2,2))
        for k in range(encoding_more[0].shape[0]):
            for j in range(encoding_more[0].shape[1]):
                encoding_more_enc[0,k,j] = more.encrypt(encoding_more[0,k,j])

        time2 = time.time()

        decoder_input_encryption_time = time2-time1

        time1 = time.time()
        decoding_plain = autodecoder_plain.predict(encoding_plain)
        time2 = time.time()
        decoding_more_enc = autodecoder_more.predict_more(encoding_more_enc)
        time3 = time.time()
        decoding_more_enc =np.nan_to_num(decoding_more_enc)

        decoder_plain_time = time2-time1
        decoder_more_time = time3-time2

        
        time1 = time.time()
        decoding_more = np.zeros((1,decoding_more_enc[0].shape[0], decoding_more_enc[0].shape[1]))
        for k in range(decoding_more_enc[0].shape[0]):
            for j in range(decoding_more_enc[0].shape[1]):
                decoding_more[0,k,j] = more.decrypt(decoding_more_enc[0][k,j])

        time2 = time.time()

        decoder_output_decryption_time = time2-time1
        decoding_accuracy_plain = mse(x_test[i], decoding_plain[0])

        decoding_accuracy_more = mse(x_test[i], decoding_more[0])
        decoding_accuracy = mse(decoding_plain[0], decoding_more[0])
        print(decoding_accuracy, decoding_accuracy_plain, decoding_accuracy_more)
        writer.writerow({'encoding_accuracy': encoding_accuracy, 'decoding_accuracy_plain': decoding_accuracy_plain , 'decoding_accuracy_more': decoding_accuracy_more, 'decoding_accuracy': decoding_accuracy, 'encoder_input_encryption_time': encoder_input_encryption_time, 'encoder_plain_time': encoder_plain_time, 'encoder_more_time': encoder_more_time, 'encoder_output_decryption_time': encoder_output_decryption_time, 'decoder_input_encryption_time': decoder_input_encryption_time, 'decoder_plain_time': decoder_plain_time, 'decoder_more_time': decoder_more_time, 'decoder_output_decryption_time': decoder_output_decryption_time })
    bar.finish()

# Check if params are transfered correctly (does number fit and so on)
# make prediction with keras and costum and compare


       

        





