import numpy as np
import random
from math import floor

from numba import njit

from activation_functions import matmul
class MoreScheme:
    def __init__(self, N):
        self.N = N        
        det = 0
        while det == 0:    
            # k = np.random.randint(self.N, size=(2,2))
            k = np.random.rand(2,2)
            det = np.linalg.det(k)
        self.key = k
        print("Generated key for N = ", self.N)

    def encrypt(self, plaintext):
        y = random.randint(floor(self.N/2),self.N)
        m = np.array([[plaintext,0],[0,y]])
        return np.matmul(self.key,np.matmul(m, np.linalg.inv(self.key)))
    def decrypt(self, cryptotext):
        p = np.matmul(np.linalg.inv(self.key),np.matmul(cryptotext, self.key))
        return p[0,0]


    def encrypt_array(self, plaintext_array):
        enc = encrypt_array_par(plaintext_array, self.key, self.N)
        return enc
    def decrypt_array(self, cryptotext):
        dec = decrypt_array_par(cryptotext, self.key)
        return dec
@njit
def encrypt_array_par(plaintext_array, key, N):
    enc = np.zeros((plaintext_array.shape[0],plaintext_array.shape[1],plaintext_array.shape[2], 2,2))
    ind = list(np.ndenumerate(plaintext_array))
    y = np.random.randint(floor(N/2), N, len(list))
    inv_k = np.linalg.inv(key)
    i = 0
    while i in range(len(ind)):
        index = ind[i][0]
        m = np.array([[plaintext_array[index],0],[0,y]])
        enc[index] = matmul(key, matmul(m, inv_k))
        i+=1
    return enc
@njit
def decrypt_array_par(cryptotext, key):
    dec = np.zeros(cryptotext.shape[0], cryptotext.shape[1], cryptotext.shape[2])
    ind = list(np.ndenumerate(dec))
    inv_k = np.linalg.inv(key)
    i = 0
    while i in range(len(ind)):
        index = ind[i][0]
        p = matmul(inv_k, matmul(cryptotext[index], key))
        dec[index] = p[0,0]
        i+=1
@njit
def matmul(x,y):
    r = np.zeros((x.shape[0], y.shape[1]))
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            for k in range(x.shape[1]):
                r[i,j] += x[i,k]*y[k,j]
    return r


