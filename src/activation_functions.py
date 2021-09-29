import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;

def relu(x):
    return np.maximum(0,x);

def relu_prime(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def sigmoid(x):
    sig = 1/(1+np.exp(-x))
    sig = np.minimum(sig, 0.999999999999999)
    sig = np.maximum(sig, 0.000000000000001)
    return sig

def sigmoid_prime(x):
    f = sigmoid(x)
    return f * (1-f)