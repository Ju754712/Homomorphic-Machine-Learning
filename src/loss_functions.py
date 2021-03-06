import numpy as np
from math import log
from mpmath import *
mp.dps=300000
from numba import njit
# loss function and its derivative
def mse(y_true, y_pred):
    try:
        x = np.mean(np.power(y_true-y_pred, 2))
    except:
        x = 0
    return x
    

def mse_prime(y_true, y_pred):
    try:
        x = 2*(y_pred-y_true)/y_true.size
    except:
        x = np.zeros(y_pred.shape)
    return x

def bce(y_true, y_pred):
    loss = np.sum(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    loss /= y_true.size
    return -loss

def bce_prime(y_true, y_pred):    
    loss = y_true*(y_pred-1)+(1-y_true)*y_pred
    loss /= y_true.size
    return loss

def mse_more(y_true, y_pred):
    ind = list(np.ndenumerate(y_pred))
    r = np.zeros((y_pred.shape[0],y_pred.shape[1],2,2))
    i = 0
    idn = np.identity(2)
    while i < len(ind):
        index = ind[i][0]
        try:
            r[(index[0],index[1])] =  matmul(t[(index[0],index[1])],t[(index[0],index[1])])
        except:
            r[(index[0],index[1])] = idn*0

        i+=1
    return r

def mse_prime_more(y_true, y_pred):
    # print(y_true.shape, y_pred.shape)
    ind = list(np.ndenumerate(y_pred))
    r = np.zeros(y_pred.shape)
    i = 0
    idn = np.identity(2)
    while i < len(ind):
        index = ind[i][0]
        try:
            r[(index[0],index[1])] = 2*(y_pred[(index[0],index[1])]-y_true[(index[1])])/(len(ind)/4)
        except:
            r[(index[0],index[1])] = idn*0

        i+=1
    return r
