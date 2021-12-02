import numpy as np
data = np.load("./src/data/finova.npy", mmap_mode='r')

data = data[0:60000]

np.save("./src/data/train.npy", data)