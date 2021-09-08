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