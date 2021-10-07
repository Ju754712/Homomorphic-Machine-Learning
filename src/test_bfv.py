import tenseal as ts


context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=8192, plain_modulus=1032193)
context.generate_galois_keys()

e1 = [1,1]
e2 = [2,2]

v1 = ts.bfv_vector(context, e1)
v2 = ts.bfv_vector(context, e2)

r = v1.dot(e2)

print(r.decrypt())

    #Adding up plain ez until plain_modulus
    #Same with substraction
    # With poly_modulus_degree = 4096 we can perform 1 plain multpilication
    # With poly_modulus_degree = 8192 we can perform 5 plain multiplication

