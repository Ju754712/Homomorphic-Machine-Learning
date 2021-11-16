import tenseal as ts

bits_scale = 26

# Create TenSEAL context
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, 31]    
)

# set the scale
context.global_scale = pow(2, bits_scale)

# galois keys are required to do ciphertext rotations
context.generate_galois_keys()

x_enc, windows_nb = ts.im2col_encoding(context, [[1],[2],[3],[4],[5],[6],[7]], 3, 1, 1)
weight = [1,2,3]

