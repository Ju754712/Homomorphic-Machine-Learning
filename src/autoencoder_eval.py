import numpy as np

import pickle

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer
from dropout_layer import DropoutLayer
from activation_functions import tanh, tanh_prime, relu, relu_prime, tanh_more, relu_more
from loss_functions import mse, mse_prime
from schemes.more import MoreScheme
import time
import csv

from progress.bar import Bar

more = MoreScheme(2)

PATH = "./src/data/train.npy"
data = np.load(PATH, mmap_mode='r')
print(data.shape)
x_test = data[0:2]
enc_plain = Network()
enc_plain.load("./src/params/autoencoder")

enc_more = Network()

# Add and encrypt first Conv Layer
enc_more.add(enc_plain.layers[0])
enc_more.layers[-1].encrypt_params_more(more)
# Add activation layer
enc_more.add(ActivationLayer(activation=relu_more, activation_prime=relu_prime))
# No Dropout Layer
# Add and encrypt second Conv Layer
enc_more.add(enc_plain.layers[3])
enc_more.layers[-1].encrypt_params_more(more)
# Add Activation layer
enc_more.add(ActivationLayer(activation=relu_more, activation_prime=relu_prime))

#Remove Decoder Levels
for i in reversed(range(5,11)):
    enc_plain.remove(i)
#Remove Dropout Layer
enc_plain.remove(2)


## DECODER BUILD


dec_plain = Network()
dec_plain.load("./src/params/autoencoder")

dec_more = Network()

# Add and encrypt first Transposed Conv Layer
dec_more.add(dec_plain.layers[5])
dec_more.layers[-1].encrypt_params_more(more)
# Add Activation Layer
dec_more.add(ActivationLayer(activation=relu_more, activation_prime=relu_prime))
# No Dropout Layer
# Add and encrypt second Transposed Conv Layer
dec_more.add(dec_plain.layers[8])
dec_more.layers[-1].encrypt_params_more(more)
# Add Tanh Activation Layer
dec_more.add(ActivationLayer(activation=tanh_more, activation_prime=tanh_prime))
dec_more.add(dec_plain.layers[10])
dec_more.layers[-1].encrypt_params_more(more)

#Remove Encoder Levels
for i in range(0,5):
    dec_plain.remove(0)
# Remove Dropout
dec_plain.remove(2)


with open('./src/csv/autoencoder_more.csv', 'w', newline='') as csvfile:
    fieldnames = ['encoder_time_for_encryption', 'encoder_time_for_evaluation_plain', 'encoder_time_for_evaluation_more', 'encoder_time_for_decryption', 'encoding_accuracy', 'decoder_time_for_encryption', 'decoder_time_for_evaluation_plain', 'decoder_time_for_evaluation_more', 'decoder_time_for_decryption', 'decoding_accuracy_plain', 'decoding_accuracy_more']
     
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    bar = Bar("Processing Samples", max = x_test.shape[0])
    for i in range(0,2):
        bar.next()
        input_more = np.zeros((1,x_test.shape[1], x_test.shape[2],2,2), dtype=np.float64)  
        time1=time.time()
        for k in range(x_test.shape[1]):                
            for m in range(x_test.shape[2]):
                input_more[0,k,m] = more.encrypt(x_test[i,k,m])
        time2 = time.time()
        encoder_time_for_encryption = time2-time1

        time1 = time.time()
        plain_encoding = enc_plain.predict([x_test[i]])
        time2 = time.time()
        more_encoding = enc_more.predict_more(input_more)
        time3 = time.time()
        encoder_time_for_evaluation_plain = time2-time1
        encoder_time_for_evaluation_more = time3-time2

        more_encoding_dec = []
        enc = np.zeros((more_encoding[0].shape[0], more_encoding[0].shape[1]))
        time1 = time.time()
        for j in range(more_encoding[0].shape[0]):
            for k in range(more_encoding[0].shape[1]):
                enc[j,k] = more.decrypt(more_encoding[0][j,k])
        time2 = time.time()
        encoder_time_for_decryption = time2-time1
        more_encoding_dec.append(enc)

    
        # Compute MSE
        more_encoding_dec[0] = np.nan_to_num(more_encoding_dec[0])
        encoding_accuracy = mse(plain_encoding[0], more_encoding_dec[0])


        input_plain_dec = np.array(plain_encoding)


        input_more_dec = np.zeros((input_plain_dec.shape[0],input_plain_dec.shape[1], input_plain_dec.shape[2],2,2), dtype=np.float64)  
        time1 = time.time()
        for k in range(input_plain_dec.shape[1]):                
            for m in range(input_plain_dec.shape[2]):
                input_more_dec[0,k,m] = more.encrypt(input_plain_dec[0,k,m])
        time2=time.time()
        decoder_time_for_encryption = time2-time1
    
        time1 = time.time()
        plain_dec = dec_plain.predict(input_plain_dec)
        time2 = time.time()
        more_dec = dec_more.predict_more(input_more_dec)
        time3 = time.time()
        decoder_time_for_evaluation_plain = time2-time1
        decoder_time_for_evaluation_more = time3-time2
        more_output_dec = []
        enc = np.zeros((more_dec[0].shape[0], more_dec[0].shape[1]))
        time1 = time.time()
        for j in range(more_dec[0].shape[0]):
            for k in range(more_dec[0].shape[1]):
                enc[j,k] = more.decrypt(more_dec[0][j,k])
        time2 = time.time()
        decoder_time_for_decryption = time2-time1
        more_output_dec.append(enc)


        more_output_dec = np.nan_to_num(more_output_dec[0])
        decoding_accuracy_plain = mse(x_test[i],plain_dec[0])
        decoding_accuracy_more = mse(x_test[i], more_output_dec[0])
        writer.writerow({'encoder_time_for_encryption': encoder_time_for_encryption, 'encoder_time_for_evaluation_plain': encoder_time_for_evaluation_plain, 'encoder_time_for_evaluation_more':encoder_time_for_evaluation_more, 'encoder_time_for_decryption': encoder_time_for_decryption, 'encoding_accuracy': encoding_accuracy, 'decoder_time_for_encryption': decoder_time_for_encryption, 'decoder_time_for_evaluation_plain': decoder_time_for_evaluation_plain, 'decoder_time_for_evaluation_more': decoder_time_for_evaluation_more, 'decoder_time_for_decryption': decoder_time_for_decryption, 'decoding_accuracy_plain': decoding_accuracy_plain, 'decoding_accuracy_more': decoding_accuracy_more})

    bar.finish()

# Mies relevante Zeiten (Encryption, Decryption, Zeit pro Layer/Vergleich mit Plaintext convolution/np um Aussagen über PErformance-Verbesserung bei vernünftiger Implementierung zu machen), Accuracy der Encodings im Vergleich mit , Accuracy der Decoding im VErgleich mit Original)