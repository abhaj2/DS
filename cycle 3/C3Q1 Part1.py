import matplotlib.pyplot as plt
import numpy as np

xaxis = np.array([2007, 2006, 2005, 2004, 2003, 2002, 2001])
yaxis = np.array([5800, 10000, 14500, 17500, 19700, 22500, 24000])
print("x-axis=> car y-axis=>year")
plt.plot(xaxis, yaxis, linestyle="dashdot", c='r', marker= '*', ms='20', mfc='green', mec='green')
plt.title("Value Depreciation", loc='left')
plt.xlabel("Car")
plt.ylabel("Year")
plt.show()
