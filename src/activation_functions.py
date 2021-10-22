import numpy as np
from scipy.linalg import expm
from mpmath import *
mp.dps=300000000

# activation function and its derivative
## Hyperbolic Tangens

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def tanh_ckks(x):
    return x

def tanh_prime_ckks(x):
    return x

def tanh_more(x):
    ind = list(np.ndenumerate(x))
    r = np.zeros(x.shape, dtype=object)
    i = 0
    while i < len(ind):
        index = ind[i][0]
        l,v = np.linalg.eig(-2*x[index])
        l_f = np.diag(np.exp(l))
        c_exp = np.matmul(v,np.matmul(l_f, np.linalg.inv(v)))
        idn = np.identity(2)
        inv = idn+c_exp
        if np.linalg.det(inv) == 0:
            r = "overflow"
        else:
            r[index] = np.matmul(2*idn, np.linalg.inv(idn+c_exp))-idn
        i+=1
    return r

def tanh_prime_more(x):
    return x
## ReLU

def relu(x):
    return np.maximum(0,x)

def relu_prime(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x


## Sigmoid

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
    r = np.zeros(x.shape, dtype=object)
    while i < len(ind):
        index = ind[i][0]
        l,v = np.linalg.eig(-x[index])
        l_f = np.diag(np.exp(l))
        c_exp = np.matmul(v,np.matmul(l_f, np.linalg.inv(v)))
        idn = np.identity(2)
        inv = idn+c_exp
        if np.linalg.det(inv) == 0:
            r = "overflow"
        else:
            r[index] = np.matmul(idn, np.linalg.inv(inv))
        i+=1
    return r

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

## Square 
def square(x):
    return np.power(x,2)

def square_prime(x):
    return 2*x

def square_bfv(x):
    return x * x

def square_prime_bfv(x):
    return 2*x

def square_ckks(x):
    return x * x

def square_prime_bfv(x):
    return 2*x

def square_more(x):
    r = np.zeros(x.shape, dtype=object)
    ind = list(np.ndenumerate(x))
    i = 0
    while i < len(ind):
        index = ind[i][0]
        r[index] = np.matmul(x[index],x[index])
        i+=1
    return r

def square_prime_more(x):
    return 2*x