import numpy as np
import random
from math import floor
class MoreScheme:
    def __init__(self, N):
        self.N = N
    def keygen(self):
        det = 0
        while det == 0:    
            k = np.random.randint(self.N, size=(2,2))
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

m = np.array([[1,0],[0,1]])

print(0 + m) 