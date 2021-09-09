from scipy import signal
import numpy as np
from conv1D_layer import Conv1DLayer, Conv1DTransposedLayer, convolution, compute_dWeights, compute_in_error
from activation_layer import ActivationLayer
from activation_functions import relu, relu_prime
from fc_layer import FCLayer
from dropout_layer import DropoutLayer
import time
from numba import cuda

print(cuda.gpus)