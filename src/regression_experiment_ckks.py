import numpy as np
from progress.bar import Bar
import time

from data_setup import  random_data
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_functions import sigmoid, sigmoid_prime, sigmoid_ckks, sigmoid_prime_ckks, square, square_ckks
import tenseal as ts
import csv




with open('lin_reg_ckks_8192_40_21_21_21_21_21_21_40_2_21.csv', 'w', newline='') as csvfile:
    fieldnames = ['Input_size','PlainOutput', 'CKKSOutput', 'CKKSOutput_Enc', 'SquarePlainOutput', 'SquareCKKSOutput', 'SigmoidPlainOutput', 'SigmoidCKKSOutput', 'Plain_Time', 'CKKS_Time', 'CKKS_Enc_Time', 'Square_Plain_Time', 'Square_CKKS_Time', 'Sigmoid_Plain_Time', 'Sigmoid_CKKS_Time' ,'Encryption_time_plain', 'Encryption_time_enc', 'Decryption_time_plain', 'Decryption_time_enc', 'Weight_enc_time']
     
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    ## Scheme Parameters

    context_ckks = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]         
          )
    context_ckks.generate_galois_keys()
    context_ckks.global_scale = 2**21


    for i in range(1,5):
        N = 2*i
        print("Input ",N)
        x_train, y_train, x_test, y_test = random_data(m=1000, n=N)
        input_size, output_size = x_train.shape[2], y_train.shape[2]
        bar = Bar('Processing Batch', max=x_train.shape[0])
        for j in range(x_train.shape[0]):
        
            # Build layer with random weights init
            fc = FCLayer(input_size, output_size)
            square_plain_layer = ActivationLayer(activation=square, activation_prime=square)
            square_ckks_layer = ActivationLayer(activation=square_ckks, activation_prime=square_ckks)
            sigmoid_plain_layer = ActivationLayer(activation=sigmoid, activation_prime=sigmoid_prime)
            sigmoid_ckks_layer = ActivationLayer(activation=sigmoid_ckks, activation_prime=sigmoid_prime_ckks)

            #Encrypt/scale layers params for different schemes
            time1 = time.time()
            fc.encrypt_params_ckks(context_ckks)
            time2= time.time()
            weights_enc_time = time2-time1



            # Encrypt input
            time1 = time.time()
            input_ckks = ts.ckks_vector(context_ckks, x_train[j][0])
            time2 = time.time()
            input_ckks_enc = np.zeros(x_train[j].shape, dtype=object)
            for k in range(x_train[j].shape[0]):
                for m in range(x_train[j].shape[1]):
                    input_ckks_enc[k,m] = ts.ckks_vector(context_ckks, [x_train[j,k,m]])
            time3 = time.time()
            enc_time_plain = time2-time1
            enc_time_enc = time3-time2

            
            # Forward-Propagation with Plaintext Weights 

            time1 = time.time()
            output = fc.forward_propagation(x_train[j])
            time2 = time.time()
            output_ckks = fc.forward_propagation_ckks(input_ckks)
            time3 = time.time()


            plain_time = time2-time1
            ckks_time = time3-time2

            # Forward-Propagation with encrypted Weights

            time1 = time.time()
            output_ckks_enc = fc.forward_propagation_ckks_encrypted(input_ckks_enc)
            time2 = time.time()

            ckks_enc_time = time2-time1


            # Square Activation Layers
            time1 = time.time()
            square_plain_res = square_plain_layer.forward_propagation(output)
            time2 = time.time()
            square_ckks_res = square_ckks_layer.forward_propagation(output_ckks)
            time3 = time.time()
          
            square_plain_time = time2 -time1
            square_ckks_time = time3 - time2


            #Sigmoid Activation Layer

            time1 = time.time()
            sigmoid_plain_res = sigmoid_plain_layer.forward_propagation(output)
            time2 = time.time()
            sigmoid_ckks_res = sigmoid_ckks_layer.forward_propagation(output_ckks)
            time3 = time.time()


            sigmoid_plain_time = time2-time1
            sigmoid_ckks_time = time3-time2

            time1 = time.time()
            ckks_output_dec = output_ckks.decrypt()[0]
            time2 = time.time()
            decryption_time_plain = time2-time1


            time1 = time.time()
            enc_ckks_output_dec = output_ckks_enc[0,0].decrypt()[0]
            time2 = time.time()
            decryption_time_enc = time2-time1




            writer.writerow({'Input_size': N,'PlainOutput':output[0,0], 'CKKSOutput':ckks_output_dec, 'CKKSOutput_Enc': enc_ckks_output_dec, 'SquarePlainOutput': square_plain_res[0,0], 'SquareCKKSOutput': square_ckks_res.decrypt()[0], 'SigmoidPlainOutput': sigmoid_plain_res[0,0], 'SigmoidCKKSOutput': square_ckks_res.decrypt()[0], 'Plain_Time': plain_time, 'CKKS_Time': ckks_time, 'CKKS_Enc_Time': ckks_enc_time, 'Square_Plain_Time': square_plain_time, 'Square_CKKS_Time': square_ckks_time, 'Sigmoid_Plain_Time': sigmoid_plain_time, 'Sigmoid_CKKS_Time': sigmoid_ckks_time ,'Encryption_time_plain': enc_time_plain, 'Encryption_time_enc': enc_time_enc, 'Decryption_time_plain': decryption_time_plain, 'Decryption_time_enc': decryption_time_enc, 'Weight_enc_time': weights_enc_time})
        
            bar.next()
        bar.finish()
