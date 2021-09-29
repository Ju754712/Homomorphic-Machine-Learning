import numpy as np

from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activation_functions import tanh, tanh_prime, sigmoid, sigmoid_prime
from loss_functions import mse, mse_prime, bce, bce_prime

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(FCLayer(2, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(3, 1))
net.add(ActivationLayer(sigmoid, sigmoid_prime))

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, batch_size=1, learning_rate=0.1)

# test
out = net.predict(x_train)
print(out)