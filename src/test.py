import numpy as np
PATH2 = "./src/data/train.npy"
PATH = "../CAE/Arrays_AE/Finova2_combined_butter.npy"
EXP = 'Finova2'
EPOCHS2TRAIN = 50
BATCHSIZE = 4
FEATURE = "combined"
TRAINON = 100 # or 'all'
MODE = 'splits'# 'splits' or 'trainon'
SAVE = False

data = np.load(PATH, mmap_mode='r+') # load data


data = data[0:4000]
with open(PATH2, 'wb') as f:
    np.save(f, data)

data = np.load(PATH2, mmap_mode='r')

print(data.shape)

