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
x_test = data[0:1]


autoencoder_plain = Network()
autoencoder_plain.load("./src/params/autoencoder_ckks")

autoencoder_ckks = Network()

for i in range(len(autoencoder_plain.layers)-1):
    autoencoder_plain.remove(-1)
print(autoencoder_plain.layers)
# Add first Conv Layer
autoencoder_ckks.add(autoencoder_plain.layers[0])



## Encryption Parameters

# controls precision of the fractional part
bits_scale = 26
print("Hello there")
print(ts)
# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]
)

#set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()

arraylength = x_test.shape[1]

with open('./src/csv/autoencoder_ckks.csv', 'w', newline='') as csvfile:
    fieldnames = ['encoding_accuracy', 'decoding_accuracy_plain', 'decoding_accuracy_ckks', 'decoding_accuracy', 'encoder_input_encryption_time', 'encoder_plain_time', 'encoder_more_time', 'encoder_output_decryption_time', 'decoder_input_encryption_time', 'decoder_plain_time', 'decoder_more_time', 'decoder_output_decryption_time' ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    # bar = Bar("Processing...", max = x_test.shape[0])
    for i in range(x_test.shape[0]):
        # bar.next()
        time_d = time.time()
        
        print("Encryption")
        x_test_ckks = np.zeros((1,x_test.shape[1], x_test.shape[2]), dtype=object)
        time1 = time.time()
        for k in range(x_test[i].shape[0]):
            for j in range(x_test[i].shape[1]):
                x_test_ckks[0,k,j] = ts.CKKSVector(context, [x_test[i,k,j]])
        time2 = time.time()
        encoder_input_encryption_time = time2-time1
        print("Encoder Input Encryption Time: ", encoder_input_encryption_time)

        time1 = time.time()
        encoding_plain = autoencoder_plain.predict(x_test[i,:,:].reshape((1,arraylength,1)))
        time2 = time.time()
        encoding_ckks_enc = autoencoder_ckks.predict_ckks(x_test_ckks)
        time3 = time.time()

        encoder_plain_time = time2-time1
        encoder_ckks_time = time3-time2
        print("Encoder Plain Time: ", encoder_plain_time)
        print("Encoder CKKS TIme: ", encoder_plain_time)



        time1 = time.time()
        encoding_more = np.zeros((1,encoding_ckks_enc[0].shape[0], encoding_ckks_enc[0].shape[1]))
        for k in range(encoding_ckks_enc[0].shape[0]):
            for j in range(encoding_ckks_enc[0].shape[1]):
                encoding_more[0,k,j] = encoding_ckks_enc[0][k,j]

        time2 = time.time()

        encoder_output_decryption_time = time2-time1
        encoding_accuracy = mse(encoding_plain[0], encoding_more[0])
        print("Encoder Output Decryption Time: ", encoder_output_decryption_time)
        print(encoding_accuracy)

        # time1 = time.time()
        # encoding_more_enc = np.zeros((encoding_more.shape[0],encoding_more.shape[1], encoding_more.shape[2],2,2))
        # for k in range(encoding_more[0].shape[0]):
        #     for j in range(encoding_more[0].shape[1]):
        #         encoding_more_enc[0,k,j] = more.encrypt(encoding_more[0,k,j])

        # time2 = time.time()

        # decoder_input_encryption_time = time2-time1

        # time1 = time.time()
        # decoding_plain = autodecoder_plain.predict(encoding_plain)
        # time2 = time.time()
        # decoding_more_enc = autodecoder_more.predict_more(encoding_more_enc)
        # time3 = time.time()
        # decoding_more_enc =np.nan_to_num(decoding_more_enc)

        # decoder_plain_time = time2-time1
        # decoder_more_time = time3-time2

        
        # time1 = time.time()
        # decoding_more = np.zeros((1,decoding_more_enc[0].shape[0], decoding_more_enc[0].shape[1]))
        # for k in range(decoding_more_enc[0].shape[0]):
        #     for j in range(decoding_more_enc[0].shape[1]):
        #         decoding_more[0,k,j] = more.decrypt(decoding_more_enc[0][k,j])

        # time2 = time.time()

        # decoder_output_decryption_time = time2-time1
        # decoding_accuracy_plain = mse(x_test[i], decoding_plain[0])

        # decoding_accuracy_more = mse(x_test[i], decoding_more[0])
        # decoding_accuracy = mse(decoding_plain[0], decoding_more[0])
        # print(x_test[i])
        # print(decoding_plain[0])
        # time_k = time.time()
        # print(time_k-time_d)
        # writer.writerow({'encoding_accuracy': encoding_accuracy, 'decoding_accuracy_plain': decoding_accuracy_plain , 'decoding_accuracy_more': decoding_accuracy_more, 'decoding_accuracy': decoding_accuracy, 'encoder_input_encryption_time': encoder_input_encryption_time, 'encoder_plain_time': encoder_plain_time, 'encoder_more_time': encoder_more_time, 'encoder_output_decryption_time': encoder_output_decryption_time, 'decoder_input_encryption_time': decoder_input_encryption_time, 'decoder_plain_time': decoder_plain_time, 'decoder_more_time': decoder_more_time, 'decoder_output_decryption_time': decoder_output_decryption_time })
    # bar.finish()

# Check if params are transfered correctly (does number fit and so on)
# make prediction with keras and costum and compare