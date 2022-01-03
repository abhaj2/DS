import numpy as np
array=np.random.randint(5,size=(2,2));
print("random array is \n ", array)
inverse=np.linalg.inv(array)
print("Inverse of the given matrix is \n", inverse)
rank=np.linalg.matrix_rank(array)
print("Rank of the matrix is \n", rank)
determinant=np.linalg.det(array)
print("Determinant of the given matrix is \n", determinant)
oned=array.flatten()
print("Transform matrix to array \n", oned)
v,w=np.linalg.eig(array)
print("Eigen values are \n",w)
print("Eigen vectors are \n",v);
