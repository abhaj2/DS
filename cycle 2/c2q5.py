import numpy as np
arr = np.arange(0,31,2)
print("First 15 even numbers are : \n", arr)
print ( "Elements from index 2 to 8 are : \n ", arr[2:8])
print ( "Elements from index 2 to 8 by slicing are : \n ", arr[slice(2,8)])
print("Last 3 elements by -ve indexing : \n " , arr[-1 -4],arr[1 -3],arr[1 -2])
print("Alternate elements in the given array are : \n ", np.arange(2,30,4))
print("alternate elements at the last 3 position :\n", arr[-3::])
