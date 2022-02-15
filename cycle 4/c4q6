
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("car.csv",names=['buying','maint','doors','persons','lug_boot','class'])


# In[ ]:


c1=data.iloc[:,-1]
X=np.unique(c1)
print(X)
C=data['class'].value_counts().sort_index()
print(C)
Y=C.values
print(Y)
plt.bar(X,Y,color='maroon',width=0.6)
plt.xlabel("Car Acceptance")
plt.ylabel("No.of cars")
plt.title("Car Evalation")
plt.show()


# In[ ]:


data['class'],class_names=pd.factorize(data['class'])
data['buying'],_=pd.factorize(data['buying'])
data['maint'],_=pd.factorize(data['maint'])
data['doors'],_=pd.factorize(data['doors'])
data['persons'],_=pd.factorize(data['persons'])
data['lug_boot'],_=pd.factorize(data['lug_boot'])
data.head()


# In[24]:


X=data.iloc[:,:-1]
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
tree1=DecisionTreeClassifier()
tree1.fit(X_train,y_train)
y_pred=tree1.predict(X_test)
count_misclassified=(y_test !=y_pred).sum()
print('Misclassified samples:',count_misclassified)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)
