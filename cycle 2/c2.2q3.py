import numpy as np

np.random.seed(42)

A = np.random.randint(0, 10, size=(5,6))

B = np.random.randint(0, 10, size=(3,3))

print("Matrix A:\n{}, shape={}\n".format(A, A.shape))

print("Matrix B:\n{}, shape={}\n".format(B, B.shape))

C = A[1:4,2:5] @ B

A[1:4,2:5] = C

print("Matrix A after submatrix multiplication:\n{}, shape={}\n".format(A, A.shape))
