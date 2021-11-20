import tenseal as ts
import numpy as np

bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree= 4*16384, #8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]    
)

# set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()


PATH = "./src/data/train.npy"
data = np.load(PATH, mmap_mode='r')
x_test = data[0]
print(x_test.shape)
x_test = np.transpose(x_test,(1,0))
print(x_test)



x_enc, windows_nb = ts.im2col_encoding(context, x_test, 1, 1, 1)
print(len(x_enc.decrypt()))
# print(windows_nb)
# kernel = [[1,2]]

# o_dec = x_enc.conv2d_im2col(kernel, windows_nb) 
# o = o_dec.decrypt()
# print(o)

