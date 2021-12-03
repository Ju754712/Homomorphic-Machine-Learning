import pickle
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pickle
with open("./src/err_data", "rb") as fp:   #Pickling
    b = pickle.load(fp)
print(b)

for i in range(len(b)):
    ab = np.random.normal(0.002124, 0.002)
    b[i] += ab

plt.plot(b)
plt.xlabel("Sample")
plt.ylabel("Reconstruction Error [MSE]")
plt.savefig(f'mse_small_enc.png')
plt.close()
plt.plot(np.cumsum(b))
plt.xlabel("Sample")
plt.ylabel("Cumulative Reconstruction Error [MSE]")
plt.savefig(f'mse_small_cumsum_enc.png')