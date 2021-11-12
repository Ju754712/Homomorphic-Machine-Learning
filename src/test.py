import numpy as np
PATH = "./src/data/train.npy"
# ERROR_SAVE_NAME = "../CAE/sr"
# EXP = 'Finova2'
# EPOCHS2TRAIN = 50
# BATCHSIZE = 4
# FEATURE = "combined"
# TRAINON = 100 # or 'all'
# MODE = 'splits'# 'splits' or 'trainon'
# SAVE = False

# data = np.load(PATH, mmap_mode='r') # load data


# data = data[4000]
# with open('./src/data/train.npy', 'wb') as f:

#     np.save(f, data)

data = np.load(PATH, mmap_mode='r')

print(data.shape)

