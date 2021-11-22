import tenseal as ts


context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
            
          )
context.generate_galois_keys()
context.global_scale = 2**21

x = [1]
w = [2]
r = [3]
x_enc = ts.CKKSVector(context, x)
w_enc= ts.CKKSVector(context, w)
r_enc= ts.CKKSVector(context, r)

for i in range(100):
    print(i)
    res = w_enc.dot(x)
    err = res-r_enc
    err_squ = err.square() * (1/2)
    w_enc -= err_squ.dot(res)
