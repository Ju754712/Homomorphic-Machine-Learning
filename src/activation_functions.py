import numpy as np
from scipy.linalg import expm
from mpmath import *
mp.dps=300000

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def relu(x):
    return np.maximum(0,x)

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

    

def sigmoid_ckks(x):
    # We use the polynomial approximation of degree 3
    # sigmoid(x) = 0.5 + 0.197 * x - 0.004 * x^3
    # from https://eprint.iacr.org/2018/462.pdf
    # which fits the function pretty well in the range [-5,5]
    return x.polyval([0.5, 0.197 , 0 , -0.004])


def sigmoid_prime_ckks(x):
    return x.polyval([0.196,0,-0.012])

def sigmoid_more(x):
    ind = list(np.ndenumerate(x))
    i = 0
    while i < len(ind):
        index = ind[i][0]
        l,v = np.linalg.eig(-x[index])
        l_f = np.diag(np.exp(l))
        c_exp = np.matmul(v,np.matmul(l_f, np.linalg.inv(v)))
        idn = np.identity(2)
        r = np.matmul(idn, np.linalg.inv(idn+c_exp))
        x[index] = r
        i+=1
    return x

def sigmoid_prime_more(x):
    f = sigmoid_more(x)
    ind = list(np.ndenumerate(x))
    i = 0

    while i < len(ind):
        index = ind[i][0]
        print(index)
        x = np.identity(2)-f[index]
        print(f[index], x)
        f[index] = np.matmul(f[index],x)
        i += 1
    return f

def square(x):
    return np.power(x,2)

def square_prime(x):
    return 2*x

def square_bfv(x):
    return x * x

def square_ckks(x):
    return x * x

def square_more(x):
    ind = list(np.ndenumerate(x))
    i = 0
    while i < len(ind):
        index = ind[i][0]
        x[index] = np.matmul(x[index],x[index])
    return x