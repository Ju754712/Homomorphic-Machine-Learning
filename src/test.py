from schemes.more import MoreScheme
import numpy as np
from mpmath import *
from activation_functions import tanh_more, tanh, sigmoid, sigmoid_more
# mp.dps=300000
a =  3
more = MoreScheme(1000)
c=more.encrypt(a)

c_a = np.zeros((1,1), dtype = object)
c_a[0,0] = c

r = tanh_more(c_a)
s = sigmoid_more(c_a)
print(more.decrypt(r[0,0]))
print(tanh(a))

# print(more.decrypt(s[0,0]))
# print(sigmoid(a))