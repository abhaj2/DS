import numpy as np
x=np.array([[4,5,8],
           [7,6,2],
           [1,2,5]])
print(x)
cube=np.power(x,3)
print("cube of the given matrix using power()  \n",cube)
cube=np.multiply(x,(x*x))
print("cube using multiply fn is \n", cube)
b=np.identity(3,dtype=int)
print("Identity matrix is \n", b)
out=np.power(x,x)
print("Displaying each element in the matrix with different powers \n", out)
y = np.arange(11,20).reshape(3,3)
print("perform the operation X^2 +2Y: \
n",np.add((np.power(x,2)),(np.multiply(y,2))))
