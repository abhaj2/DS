import numpy as np

array_2d=np.array([[complex(1,2),complex(2,3),complex(3,4)],[complex(4,5),complex(5,6),complex(6,7)]])
print(array_2d)

print("the no. of rows and columns :",array_2d.shape)
print("dimension of an array",array_2d.ndim)

print("reshape the same array to 3x2",array_2d.reshape(3,2))
