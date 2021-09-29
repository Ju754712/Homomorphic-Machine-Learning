import matplotlib.pyplot as plt
from data_setup import heart_disease_data, titanic_data
from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_functions import sigmoid, sigmoid_prime, tanh, tanh_prime
from loss_functions import bce, bce_prime, mse, mse_prime

x_train, y_train, x_test, y_test = heart_disease_data()

input_size, output_size = x_train.shape[2], y_train.shape[2]


# # BUILDING NETWORK

net = Network()

net.add(FCLayer(input_size, output_size))
net.add(ActivationLayer(activation=sigmoid, activation_prime=sigmoid_prime))
net.use(loss=bce, loss_prime=bce_prime)

net.save("logreg_raw")

#PLAIN TRAINING NETWORK

net.fit(x_train, y_train, epochs=100, batch_size = 1, learning_rate=0.5)

net.save("logreg_trained")

print(net.layers[0].weights)

out = net.predict(x_test[0:10])

print(out, y_test[0:10])