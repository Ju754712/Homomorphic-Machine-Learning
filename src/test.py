import tenseal as ts
import numpy as np

context = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=4096, plain_modulus=1032193)
context.generate_galois_keys()


plain_vector = [60, 66, 73, 81, 90]

encrypted_vector = ts.bfv_vector(context, plain_vector)

weights = np.array([-50,-50,-50,-50,-50])
bias = np.random.rand(2)

result = encrypted_vector.dot(weights) 
print(result.decrypt())


weights = np.random.rand(5, 5) - 0.5

print(weights)