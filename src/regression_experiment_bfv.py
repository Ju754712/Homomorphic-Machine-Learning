import numpy as np
from progress.bar import Bar
import time

from data_setup import heart_disease_data, random_data
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_functions import square, square_bfv, square_prime, square_prime_bfv

import tenseal as ts
import csv
#enable batching, we need to set the plain_modulus to be a prime number
    #congruent to 1 modulo 2*poly_modulus_degree.

# Generate data 

x_train, y_train, x_test, y_test = random_data(m=1000, n=3)
input_size, output_size = x_train.shape[2], y_train.shape[2]
print(input_size, output_size)

with open('lin_reg_bfv_16384_1099510054913.csv', 'w', newline='') as csvfile:
    fieldnames = ['Input_size', 'Input_scale', 'Param_scale' ,'PlainOutput', 'BFVOutput', 'BFVOutput_Enc', 'SquarePlainOutput', 'SquareBFVOutput', 'Plain_Time', 'BFV_Time', 'BFV_Enc_Time', 'Square_Plain_Time', 'Square_BFV_Time' ,'Encryption_time_plain', 'Encryption_time_enc', 'Decryption_time_plain', 'Decryption_time_enc', 'Weight_enc_time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for s in range(1,4):
        for l in range(1,4):
            for o in range(1,5):
                N = 2*o
                # BFV Setup
                context_bfv = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=16384, plain_modulus=1099510054913)
                context_bfv.generate_galois_keys()
                input_scale = 10**s
                param_scale = 10**l
                scale = input_scale * param_scale
                print(N, input_scale, param_scale)
                bar = Bar('Processing Batch', max=x_train.shape[0])
                x_train, y_train, x_test, y_test = random_data(m=1000, n=N)
                input_size, output_size = x_train.shape[2], y_train.shape[2]
                for j in range(x_train.shape[0]):

        
                    # Build layer with random weights init
                    fc = FCLayer(input_size, output_size)
                    square_plain_layer = ActivationLayer(activation=square, activation_prime=square)
                    square_bfv_layer = ActivationLayer(activation=square_bfv, activation_prime=square_bfv)


                    #Encrypt/scale layers params for different schemes
                    fc.scale_params_bfv(param_scale)
 
                    # Encrypt/scale input
                    input_scaled = (x_train[j]*input_scale).astype(int)
                    time1 = time.time()
                    input_bfv = ts.bfv_vector(context_bfv, input_scaled[0])
                    time2 = time.time()
                    input_bfv_enc = np.zeros(x_train[j].shape, dtype=object)
                    for k in range(x_train[j].shape[0]):
                        for m in range(x_train[j].shape[1]):
                            input_bfv_enc[k,m] = ts.bfv_vector(context_bfv, [input_scaled[k,m]])
                    time3 = time.time()

                    encryption_time_plain = time2-time1
                    encryption_time_enc = time3-time2
                   
                    # Forward-Propagation with Plaintext Weights 

                    time1 = time.time()
                    output = fc.forward_propagation(x_train[j])
                    time2 = time.time()
                    output_bfv = fc.forward_propagation_bfv(input_bfv)
                    time3 = time.time()


                    plain_time = time2-time1
                    bfv_time = time3-time2
                    time1 = time.time()
                    fc.encrypt_params_bfv(context_bfv)
                    time2 = time.time()
                    weight_enc_time = time2-time1

                    # Forward-Propagation with encrypted Weights

                    time1 = time.time()
                    output_bfv_enc = fc.forward_propagation_bfv_encrypted(input_bfv_enc)
                    time2 = time.time()

                    bfv_enc_time = time2-time1


                    # Square Activation Layers
                    time1 = time.time()
                    square_plain_res = square_plain_layer.forward_propagation(output)
                    time2 = time.time()
                    square_bfv_res = square_bfv_layer.forward_propagation(output_bfv)
                    time3 = time.time()
            
                    square_plain_time = time2 -time1
                    square_bfv_time = time3 - time2

                    time1 = time.time()
                    bfv_output_dec = output_bfv[0,0].decrypt()[0]/scale
                    time2 = time.time()
                    decryption_time_plain = time2-time1
                    time1 = time.time()
                    enc_bfv_output_dec = output_bfv_enc[0,0].decrypt()[0]/scale
                    time2 = time.time()
                    decryption_time_enc = time2-time1




                    writer.writerow({'Input_size': N, 'Input_scale': input_scale, 'Param_scale': param_scale, 'PlainOutput': output[0,0], 'BFVOutput': bfv_output_dec, 'BFVOutput_Enc': enc_bfv_output_dec, 'SquarePlainOutput': square_plain_res[0,0], 'SquareBFVOutput': square_bfv_res[0,0].decrypt()[0]/scale**2,  'Plain_Time': plain_time, 'BFV_Time': bfv_time, 'BFV_Enc_Time': bfv_enc_time, 'Square_Plain_Time': square_plain_time, 'Square_BFV_Time': square_bfv_time ,'Encryption_time_plain': encryption_time_plain, 'Encryption_time_enc': encryption_time_enc, 'Decryption_time_plain': decryption_time_plain, 'Decryption_time_enc': decryption_time_enc,'Weight_enc_time': weight_enc_time})
                
                    bar.next()
                bar.finish()

