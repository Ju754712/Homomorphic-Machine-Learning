import tenseal as ts
# Setup TenSEAL context
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
          )

# plain_vector = [60, 66, 73, 81, 90]
# encrypted_vector = ts.bfv_vector(context, plain_vector)

# add_result = encrypted_vector + [1, 2, 3, 4, 5]
# print(add_result.decrypt())

