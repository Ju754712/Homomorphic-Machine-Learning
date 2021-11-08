import numpy as np
from math import log
from mpmath import *
mp.dps=300000
# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));
    

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

def bce(y_true, y_pred):
    loss = np.sum(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    loss /= y_true.size
    return -loss

def bce_prime(y_true, y_pred):    
    loss = y_true*(y_pred-1)+(1-y_true)*y_pred
    loss /= y_true.size
    return loss

def mse_more(y_true, y_pred):
    return 0

def mse_prime_more(y_true, y_pred):
    return 2/y_true.size*(y_pred-y_true)