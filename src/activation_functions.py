import numpy as np
from scipy.linalg import expm
from mpmath import *
from progress.bar import Bar
mp.dps=300000000
from numba import njit

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
@njit
def tanh_more(x):

    ind = list(np.ndenumerate(x))
    r = np.zeros((x.shape[0],x.shape[1],2,2))
    i = 0
    idn = np.identity(2)
    while i < len(ind):
        index = ind[i][0]
        try:
            l,v = np.linalg.eig(-2*x[(index[0],index[1])])
            l_f = np.diag(np.exp(l))
            c_exp = matmul(v,matmul(l_f, np.linalg.inv(v)))
            inv = idn+c_exp
            if np.linalg.det(inv) == 0:
                r[(index[0],index[1])] = idn*0 # "overflow"
            else:
                r[(index[0],index[1])] = matmul(2*idn, np.linalg.inv(idn+c_exp))-idn
        except:
            r[(index[0],index[1])] = idn*0

        i+=1
    return r
@njit
def tanh_prime_more(x):
    t = tanh_more(x)
    ind = list(np.ndenumerate(x))
    r = np.zeros((x.shape[0],x.shape[1],2,2))
    i = 0
    idn = np.identity(2)
    while i < len(ind):
        index = ind[i][0]
        
        try:
            
            r[(index[0],index[1])] = idn - matmul(t[(index[0],index[1])],t[(index[0],index[1])])
        except:
            r[(index[0],index[1])] = idn*0

        i+=1
        
    return r

def relu(x):
    return np.maximum(0,x)

def relu_approx(x):
    #return -0.012*x**2 + 0.394
    return x**2 + x
@njit
def relu_more(x):
    ind = list(np.ndenumerate(x))
    i = 0
    idn = np.identity(2)
    r = np.zeros((x.shape[0],x.shape[1],2,2))
    while i < len(ind):
        try:
            index = ind[i][0]
            q = matmul(x[(index[0],index[1])], x[(index[0], index[1])])
            l,v = np.linalg.eig(q)

            l_f = np.diag(np.sqrt(l))
            c= matmul(v,matmul(l_f, np.linalg.inv(v)))

            r[(index[0],index[1])] = 1/2*(x[(index[0],index[1])]+c)
        except: 
            r[(index[0],index[1])] = idn*0
        i +=1
    return r




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

def sigmoid_approx(x):
    try:
        r = -0.004 * np.power(x,3) + 0.197*x +0.5
    except:
        r = np.zeros(x.shape)
    return r

def sigmoid_approx_prime(x):
    try:
        r = -0.012*np.power(x,2)+0.197
    except:
        r = np.zeros(x.shape)
    return r

@njit
def sigmoid_approx_more(x):
    ind = list(np.ndenumerate(x))
    i = 0
    r = np.zeros((x.shape[0],x.shape[1],2,2))
    idn = np.identity(2)
    while i < len(ind):
        index = ind[i][0]
        l,v = np.linalg.eig(x[(index[0],index[1])])
        l_f = np.diag(np.power(l,3))
        x_power= matmul(v,matmul(l_f, np.linalg.inv(v)))
        r[(index[0], index[1])] = -0.004 * x_power+ 0.197 *x[(index[0], index[1])]+0.5*idn
        i+=1
    return r


@njit
def sigmoid_approx_prime_more(x):
    ind = list(np.ndenumerate(x))
    i = 0
    r = np.zeros((x.shape[0],x.shape[1],2,2))
    while i < len(ind):
        index = ind[i][0]
        r[(index[0], index[1])] = -0.012 * matmul(x[(index[0], index[1])],x[(index[0], index[1])]) + 0.197 *x[(index[0], index[1])]
        i+=1
    return r

def sigmoid_approx_ckks(x):
    # We use the polynomial approximation of degree 3
    # sigmoid(x) = 0.5 + 0.197 * x - 0.004 * x^3
    # from https://eprint.iacr.org/2018/462.pdf
    # which fits the function pretty well in the range [-5,5]
    r = np.zeros(x.shape, dtype=object)
    ind = list(np.ndenumerate(x))
    i = 0
    while i < len(ind):
        index = ind[i][0]
        r[index] = x[index].polyval([0.5, 0.197 , 0 , -0.004])
        i+=1
    return r


def sigmoid_prime_ckks(x):
    return x.polyval([0.196,0,-0.012])

@njit
def sigmoid_more(x):
    ind = list(np.ndenumerate(x))
    i = 0
    idn = np.identity(2)
    r = np.zeros((x.shape[0],x.shape[1],2,2))
    while i < len(ind):
        index = ind[i][0]
        l,v = np.linalg.eig(-x[(index[0], index[1])])
        l_f = np.diag(np.exp(l))
        c_exp = matmul(v,matmul(l_f, np.linalg.inv(v)))

        inv = idn+c_exp
        if np.linalg.det(inv) == 0:
            r[(index[0], index[1])] = idn*0
        else:
            r[(index[0], index[1])] = matmul(idn, np.linalg.inv(inv))
        i+=1
    return r

@njit
def sigmoid_prime_more(x):
    f = sigmoid_more(x)
    r = np.zeros(f.shape)
    ind = list(np.ndenumerate(x))
    i = 0
    idn = np.identity(2)
    while i < len(ind):
        index = ind[i][0]
        x = idn-f[(index[0], index[1])]
        r[(index[0], index[1])] = matmul(f[(index[0], index[1])],x)
        i += 1
    return r

## Square 
def square(x):
    return np.power(x,2)+x

def square_prime(x):
    return 2*x+1

def square_bfv(x):
    return x * x

def square_prime_bfv(x):
    return 2*x

def square_ckks(x):
    r = np.zeros(x.shape, dtype=object)
    ind = list(np.ndenumerate(x))
    i = 0
    while i < len(ind):
        index = ind[i][0]
        r[index] = x[index].square()
        i+=1
    return r

def square_prime_bfv(x):
    return 2*x
@njit
def square_more(x):
    r = np.zeros((x.shape[0],x.shape[1],2,2))
    ind = list(np.ndenumerate(x))
    i = 0
    while i < len(ind):
        index = ind[i][0]
        r[(index[0],index[1])] = matmul(x[(index[0],index[1])],x[(index[0],index[1])])+x[(index[0],index[1])]
        i+=1
    return r
@njit
def square_prime_more(x):
    r = np.zeros((x.shape[0],x.shape[1],2,2))
    ind = list(np.ndenumerate(x))
    idn = np.identity(2)
    i = 0
    while i < len(ind):
        index = ind[i][0]
        r[(index[0],index[1])] = 2*x[(index[0],index[1])]+idn
        i+=1
    return r

@njit
def matmul(x,y):
    r = np.zeros((x.shape[0], y.shape[1]))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            for k in range(x.shape[1]):
                r[i,j] += x[i,k]*y[k,j]
    return r