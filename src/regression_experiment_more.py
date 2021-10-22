import numpy as np
from progress.bar import Bar
import time

from data_setup import random_data
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_functions import sigmoid, sigmoid_prime, sigmoid_more, sigmoid_prime_more, square, square_more, tanh, tanh_prime, tanh_more

from schemes.more import MoreScheme
import csv





with open('lin_reg_more.csv', 'w', newline='') as csvfile:
    fieldnames = ['N', 'Input_size','PlainOutput', 'MoreOutput', 'MoreOutput_Enc', 'SquarePlainOutput', 'SquareMoreOutput', 'SigmoidPlainOutput', 'SigmoidMoreOutput', 'TanhPlainOutput', 'TanhMoreOutput', 'Plain_Time', 'More_Time', 'More_Enc_Time', 'Square_Plain_Time', 'Square_More_Time', 'Sigmoid_Plain_Time', 'Sigmoid_More_Time', 'Tanh_Plain_Time', 'Tanh_More_Time' ,'Encryption_time', 'Decryption_time', 'Weight_enc_time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    n_arr = [2,5,10,50,100,150,200,400]
    
    for N in n_arr:

        ## Define Cryptosystems   
        more = MoreScheme(N)
        print (more.key)
        for i in range(1,5):
            input_size = 2*i
            x_train, y_train, x_test, y_test = random_data(m=1000, n=input_size)
            output_size = y_train.shape[2]
            print("Input Size: ",input_size)
            bar = Bar('Processing Batch', max=x_train.shape[0])
            for j in range(x_train.shape[0]):
        
                # Build layer with random weights init
                fc = FCLayer(input_size, output_size)
                square_plain_layer = ActivationLayer(activation=square, activation_prime=square)
                square_more_layer = ActivationLayer(activation=square_more, activation_prime=square_more)
                sigmoid_plain_layer = ActivationLayer(activation=sigmoid, activation_prime=sigmoid_prime)
                sigmoid_more_layer = ActivationLayer(activation=sigmoid_more, activation_prime=sigmoid_prime_more)
                tanh_plain_layer = ActivationLayer(activation=tanh, activation_prime=tanh_prime)
                tanh_more_layer = ActivationLayer(activation=tanh_more, activation_prime=tanh_more)

                #Encrypt/scale layers params for different schemes
                time1 = time.time()
                fc.encrypt_params_more(more)    
                time2 = time.time()

                weight_enc_time = time2 -time1


                # Encrypt input
        
                input_more = np.zeros(x_train[j].shape, dtype=object)
                time1 = time.time()
                for k in range(x_train[j].shape[0]):
                    for m in range(x_train[j].shape[1]):
                        input_more[k,m] = more.encrypt(x_train[j,k,m])
            
                time2 = time.time()

                encryption_time = time2-time1
                # Forward-Propagation with Plaintext Weights 

                time1 = time.time()
                output = fc.forward_propagation(x_train[j])
                time2 = time.time()
                output_more = fc.forward_propagation_more(input_more)
                time3 = time.time()

                plain_time = time2-time1
                more_time = time3-time2

                # Forward-Propagation with encrypted Weights

                time1 = time.time()
                output_more_enc = fc.forward_propagation_more_encrypted(input_more)
                time2 = time.time()
                more_enc_time = time2-time1

                # Square Activation Layers
                time1 = time.time()
                square_plain_res = square_plain_layer.forward_propagation(output)
                time2 = time.time()
                square_more_res = square_more_layer.forward_propagation(output_more)
                time3 = time.time()
            
                square_plain_time = time2 -time1
                square_more_time = time3 - time2

                #Sigmoid Activation Layer

                time1 = time.time()
                sigmoid_plain_res = sigmoid_plain_layer.forward_propagation(output)
                time2 = time.time()
                sigmoid_more_res = sigmoid_more_layer.forward_propagation(output_more)
                time3 = time.time()

                sigmoid_plain_time = time2-time1
                sigmoid_more_time = time3-time2

                #Tanh Activation Layer
                time1 = time.time()
                tanh_plain_res = tanh_plain_layer.forward_propagation(output)
                time2 = time.time()
                tanh_more_res = tanh_more_layer.forward_propagation(output_more)
                time3 = time.time()

                tanh_plain_time = time2-time1
                tanh_more_time = time3 - time2

                time1 = time.time()
                o_more = more.decrypt(output_more[0,0])
                time2 = time.time()
                decryption_time = time2-time1

                if sigmoid_more_res != "overflow":
                    sigmoid_more_res = more.decrypt(sigmoid_more_res[0,0])
                else:
                    print("overflow")
                if tanh_more_res != "overflow":
                    tanh_more_res = more.decrypt(tanh_more_res[0,0])
                else:
                    print("overflow")



                writer.writerow({'N': N, 'Input_size': input_size, 'PlainOutput': output[0,0], 'MoreOutput': o_more, 'MoreOutput_Enc': more.decrypt(output_more_enc[0,0]), 'SquarePlainOutput': square_plain_res[0,0], 'SquareMoreOutput': more.decrypt(square_more_res[0,0]), 'SigmoidPlainOutput': sigmoid_plain_res[0,0], 'SigmoidMoreOutput': sigmoid_more_res, 'TanhPlainOutput': tanh_plain_res[0,0], 'TanhMoreOutput':tanh_more_res, 'Plain_Time': plain_time, 'More_Time': more_time, 'More_Enc_Time': more_enc_time, 'Square_Plain_Time': square_plain_time, 'Square_More_Time': square_more_time, 'Sigmoid_Plain_Time': sigmoid_plain_time, 'Sigmoid_More_Time': sigmoid_more_time, 'Tanh_Plain_Time': tanh_plain_time, 'Tanh_More_Time': tanh_more_time ,'Encryption_time': encryption_time, 'Decryption_time': decryption_time, 'Weight_enc_time': weight_enc_time})

                bar.next()
            bar.finish()