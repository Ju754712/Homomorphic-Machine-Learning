import numpy as np
from progress.bar import Bar
import time

from data_setup import heart_disease_data, random_data
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_functions import sigmoid, sigmoid_prime, sigmoid_ckks, sigmoid_prime_ckks, sigmoid_more, sigmoid_prime_more, square, square_bfv, square_ckks, square_more

import tenseal as ts
from schemes.more import MoreScheme
import csv
#enable batching, we need to set the plain_modulus to be a prime number
    #congruent to 1 modulo 2*poly_modulus_degree.

# Generate data 

x_train, y_train, x_test, y_test = random_data(m=1000, n=3)
input_size, output_size = x_train.shape[2], y_train.shape[2]
print(input_size, output_size)



with open('lin_reg.csv', 'w', newline='') as csvfile:
    fieldnames = ['PlainOutput', 'BFVOutput', 'CKKSOutput', 'MOREOutput', 'CKKSOutput_Enc', 'MOREOutput_Enc', 'BFVOutput_Enc', 'Plain_Time', 'BFV_Time', 'CKKS_Time', 'MORE_Time', 'BFV_Enc_Time', 'CKKS_Enc_Time', 'MORE_Enc_Time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    ## Define Cryptosystems

    # BFV Scheme


    context_bfv = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=16384, plain_modulus=1099510054913)
    context_bfv.generate_galois_keys()
    input_scale = 100
    param_scale = 100
    scale = input_scale * param_scale

    # CKKS Scheme

    context_ckks = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]         
          )
    context_ckks.generate_galois_keys()
    context_ckks.global_scale = 2**21

    # More scheme

    v1 = ts.bfv_vector(context_bfv, [1])
    
    more = MoreScheme(N=40000)

    # Iterate over whole data set for K iterations
    K = 1

    for i in range(K):
    # for i in range(1):
        print("Iteration: ",i)
        #for j in range(x_train.shape[0]):
        bar = Bar('Processing Batch', max=x_train.shape[0])
        for j in range(x_train.shape[0]):
        # for j in range(10):
        
            # Build layer with random weights init
            fc = FCLayer(input_size, output_size)
            sq_plain = ActivationLayer(activation=square, activation_prime=square)
            sq_bfv = ActivationLayer(activation=square_bfv, activation_prime=square_bfv)
            sq_ckks = ActivationLayer(activation=square_ckks, activation_prime=square_ckks)
            sq_more = ActivationLayer(activation=square_more, activation_prime=square_more)
            a_ckks = ActivationLayer(activation=sigmoid_ckks, activation_prime=sigmoid_prime_ckks)
            a_more = ActivationLayer(activation=sigmoid_more, activation_prime=sigmoid_prime_more)

            #Encrypt/scale layers params for different schemes
            fc.scale_params_bfv(param_scale)

            fc.encrypt_params_ckks(context_ckks)

            fc.encrypt_params_more(more)    


            # Encrypt/scale input
            input_scaled = (x_train[j]*input_scale).astype(int)
            input_bfv = ts.bfv_vector(context_bfv, input_scaled[0])
            input_bfv_enc = np.zeros(x_train[j].shape, dtype=object)
            for k in range(x_train[j].shape[0]):
                for m in range(x_train[j].shape[1]):
                    input_bfv_enc[k,m] = ts.bfv_vector(context_bfv, [input_scaled[k,m]])

            input_ckks = ts.ckks_vector(context_ckks, x_train[j][0])
            input_ckks_enc = np.zeros(x_train[j].shape, dtype=object)
            for k in range(x_train[j].shape[0]):
                for m in range(x_train[j].shape[1]):
                    input_ckks_enc[k,m] = ts.ckks_vector(context_ckks, [x_train[j,k,m]])

        
            input_more = np.zeros(x_train[j].shape, dtype=object)
            for k in range(x_train[j].shape[0]):
                for m in range(x_train[j].shape[1]):
                    input_more[k,m] = more.encrypt(x_train[j,k,m])
            
            # Forward-Propagation with Plaintext Weights 

            time1 = time.time()
            output = fc.forward_propagation(x_train[j])
            time2 = time.time()
            output_bfv = fc.forward_propagation_bfv(input_bfv)
            time3 = time.time()
            output_ckks = fc.forward_propagation_ckks(input_ckks)
            time4 = time.time()
            output_more = fc.forward_propagation_more(input_more)
            time5 = time.time()

            plain_time = time2-time1
            bfv_time = time3-time2
            ckks_time = time4-time3
            more_time = time5-time4

            fc.encrypt_params_bfv(context_bfv)

            # Forward-Propagation with encrypted Weights

            time1 = time.time()
            output_bfv_enc = fc.forward_propagation_bfv_encrypted(input_bfv_enc)
            time2 = time.time()
            output_ckks_enc = fc.forward_propagation_ckks_encrypted(input_ckks_enc)
            time3 = time.time()
            output_more_enc = fc.forward_propagation_more_encrypted(input_more)
            time4 = time.time()
            bfv_enc_time = time2-time1
            ckks_enc_time = time3-time2
            more_enc_time = time4-time3

            # Activation Layer
            # time1 = time.time()
            # sq_plain = sq_plain.forward_propagation(output)
            # time2 = time.time()
            # sq_bfv = sq_bfv.forward_propagation(output_bfv)
            # time3 = time.time()
            # sig_ckks = a_ckks.forward_propagation(output_ckks)
            # time4 = time.time()
            # sig_more = a_more.forward_propagation(output_more)
            # time5 = time.time()

            # plain_act_time = time2 - time1
            # bfv_act_time = time3-time2


            # print("Output: ", output[0,0])
            # print("Output BFV: ", output_bfv[0,0].decrypt()[0]/scale)
            # print("Output CKKS: ", output_ckks.decrypt()[0])
            # print("Output MORE: ", more.decrypt(output_more[0,0]))
            # print("Output encrypted BFV:")
            # for i in range(output_bfv_enc.shape[0]):
            #     for j in range(output_bfv_enc.shape[1]):
            #         print(output_bfv_enc[i,j].decrypt()[0]/scale)

            # # print("Output Square: ", sq_plain)
            # # print("Output Square BFV: ", sq_bfv[0][0].decrypt()[0]/(scale**2))
            # print("Output encrypted CKKS:")
            # for i in range(output_ckks_enc.shape[0]):
            #     for j in range(output_ckks_enc.shape[1]):
            #         print(output_ckks_enc[i,j].decrypt()[0])

            # print("Output encrypted MORE:")
            # for i in range(output_more_enc.shape[0]):
            #     for j in range(output_more_enc.shape[1]):
            #         print(more.decrypt(output_more_enc[i,j]))


            writer.writerow({'PlainOutput': output[0,0], 'BFVOutput': output_bfv[0,0].decrypt()[0]/scale, 'CKKSOutput': output_ckks.decrypt()[0], 'MOREOutput': more.decrypt(output_more[0,0]), 'BFVOutput_Enc': output_bfv_enc[0,0].decrypt()[0]/scale, 'CKKSOutput_Enc': output_ckks_enc[0,0].decrypt()[0], 'MOREOutput_Enc': more.decrypt(output_more_enc[0,0]), 'Plain_Time': plain_time, 'BFV_Time': bfv_time, 'CKKS_Time': ckks_time, 'MORE_Time': more_time, 'BFV_Enc_Time': bfv_enc_time, 'CKKS_Enc_Time': ckks_enc_time, 'MORE_Enc_Time': more_enc_time})

            bar.next()
        bar.finish()

