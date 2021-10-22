import numpy as np
from progress.bar import Bar
import time

from data_setup import heart_disease_data, random_data
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_functions import sigmoid, sigmoid_prime, sigmoid_ckks, sigmoid_prime_ckks, sigmoid_more, sigmoid_prime_more, square, square_bfv, square_ckks, square_more, tanh, tanh_prime, tanh_more

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
        # bar = Bar('Processing Batch', max=x_train.shape[0])
        for j in range(x_train.shape[0]):
        # for j in range(10):
        
            # Build layer with random weights init
            fc = FCLayer(input_size, output_size)
            square_plain_layer = ActivationLayer(activation=square, activation_prime=square)
            square_bfv_layer = ActivationLayer(activation=square_bfv, activation_prime=square_bfv)
            square_ckks_layer = ActivationLayer(activation=square_ckks, activation_prime=square_ckks)
            square_more_layer = ActivationLayer(activation=square_more, activation_prime=square_more)
            sigmoid_plain_layer = ActivationLayer(activation=sigmoid, activation_prime=sigmoid_prime)
            sigmoid_ckks_layer = ActivationLayer(activation=sigmoid_ckks, activation_prime=sigmoid_prime_ckks)
            sigmoid_more_layer = ActivationLayer(activation=sigmoid_more, activation_prime=sigmoid_prime_more)
            tanh_plain_layer = ActivationLayer(activation=tanh, activation_prime=tanh_prime)
            tanh_more_layer = ActivationLayer(activation=tanh_more, activation_prime=tanh_more)

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

            # Square Activation Layers
            time1 = time.time()
            square_plain_res = square_plain_layer.forward_propagation(output)
            time2 = time.time()
            square_bfv_res = square_bfv_layer.forward_propagation(output_bfv)
            time3 = time.time()
            square_ckks_res = square_ckks_layer.forward_propagation(output_ckks)
            time4 = time.time()
            square_more_res = square_more_layer.forward_propagation(output_more)
            time5 = time.time()
            
            square_plain_time = time2 -time1
            square_bfv_time = time3 - time2
            square_ckks_time = time4 - time3
            square_more_time = time5 - time4

            #Sigmoid Activation Layer

            time1 = time.time()
            sigmoid_plain_res = sigmoid_plain_layer.forward_propagation(output)
            time2 = time.time()
            sigmoid_ckks_res = sigmoid_ckks_layer.forward_propagation(output_ckks)
            time3 = time.time()
            sigmoid_more_res = sigmoid_more_layer.forward_propagation(output_more)#sigmoid_more_layer.forward_propagation(output_more)
            time4 = time.time()

            sigmoid_plain_time = time2-time1
            sigmoid_ckks_time = time3-time2
            sigmoid_more_time = time4-time3

            #Tanh Activation Layer
            time1 = time.time()
            tanh_plain_res = tanh_plain_layer.forward_propagation(output)
            time2 = time.time()
            tanh_more_res = tanh_more_layer.forward_propagation(output_more)
            time3 = time.time()

            tanh_plain_time = time2-time1
            tanh_more_time = time3 - time2


            print("Computed Plain Output: ", output[0,0], " in ", plain_time, " seconds")
            print("Computed BFV Output: ", output_bfv[0,0].decrypt()[0]/scale, " in ", bfv_time, " seconds")
            print("Computed CKKS Output: ", output_ckks.decrypt()[0], " in ", ckks_time, " seconds")
            print("Computed MORE Output: ")
            for i in range(output_more.shape[0]):
                for j in range(output_more.shape[1]):
                    print(more.decrypt(output_more[i,j]))
            print(" in ", more_time, " seconds")
            print("Computed encrypted BFV Output:")
            for i in range(output_bfv_enc.shape[0]):
                for j in range(output_bfv_enc.shape[1]):
                    print(output_bfv_enc[i,j].decrypt()[0]/scale)
            print("in ",bfv_enc_time, " seconds")

            print("Computed  encrypted CKKS Output:")
            for i in range(output_ckks_enc.shape[0]):
                for j in range(output_ckks_enc.shape[1]):
                    print(output_ckks_enc[i,j].decrypt()[0])
            print("in ",ckks_enc_time, " seconds")
            print("Computed  encrypted MORE Output:")
            for i in range(output_more_enc.shape[0]):
                for j in range(output_more_enc.shape[1]):
                    print(more.decrypt(output_more_enc[i,j]))
            print("in ",more_enc_time, " seconds")

            
            print("Computed Plain Square Output: ", square_plain_res[0,0], " in ", square_plain_time, " seconds")
            print("Computed BFV Output: ", square_bfv_res[0,0].decrypt()[0]/scale**2, " in ", square_bfv_time, " seconds")
            print("Computed CKKS Output: ", square_ckks_res.decrypt()[0], " in ", square_ckks_time, " seconds")
            print("Computed MORE Output: ")
            for i in range(square_more_res.shape[0]):
                for j in range(square_more_res.shape[1]):
                    print(more.decrypt(square_more_res[i,j]))
            print(" in ", square_more_time, " seconds")

            print("Computed Plain Sigmoid Output: ", sigmoid_plain_res[0,0], " in ", sigmoid_plain_time, " seconds")
            print("Computed MORE Sigmoid Output: ")
            for i in range(sigmoid_more_res.shape[0]):
                for j in range(sigmoid_more_res.shape[1]):
                    print(more.decrypt(sigmoid_more_res[i,j]))
            print(" in ", sigmoid_more_time, " seconds")

            print("Computed Plain Tanh Output: ", tanh_plain_res[0,0], " in ", tanh_plain_time, " seconds")
            print("Computed MORE Tanh Output: ")
            for i in range(tanh_more_res.shape[0]):
                for j in range(tanh_more_res.shape[1]):
                    print(more.decrypt(tanh_more_res[i,j]))
            print(" in ", tanh_more_time, " seconds")

            #+writer.writerow({'PlainOutput': output[0,0], 'BFVOutput': output_bfv[0,0].decrypt()[0]/scale, 'CKKSOutput': output_ckks.decrypt()[0], 'MOREOutput': more.decrypt(output_more[0,0]), 'BFVOutput_Enc': output_bfv_enc[0,0].decrypt()[0]/scale, 'CKKSOutput_Enc': output_ckks_enc[0,0].decrypt()[0], 'MOREOutput_Enc': more.decrypt(output_more_enc[0,0]), 'Plain_Time': plain_time, 'BFV_Time': bfv_time, 'CKKS_Time': ckks_time, 'MORE_Time': more_time, 'BFV_Enc_Time': bfv_enc_time, 'CKKS_Enc_Time': ckks_enc_time, 'MORE_Enc_Time': more_enc_time})

        #     bar.next()
        # bar.finish()

