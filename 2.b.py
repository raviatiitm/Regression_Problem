#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random


# In[9]:


df=pd.read_csv('A2Q2Data_train.csv',header=None)
df1=pd.read_csv('A2Q2Data_test.csv',header=None)


# In[10]:


traindata=np.array(df)
testdata=np.array(df1)


# In[11]:


XT=traindata[:,0:100]


# In[12]:


X=XT.T


# In[13]:


Y=traindata[:,100].T


# In[14]:


Xtest=testdata[:,0:100]
Ytest=testdata[:,100]


# In[15]:


XXT=np.matmul(X,XT)


# In[16]:


XY=np.matmul(X,Y)


# In[17]:


W=np.zeros((100))


# In[18]:


Wml=np.matmul(np.linalg.inv(XXT),XY)


# In[19]:


def derivative(w):
    ans=2*(np.matmul(XXT,w)) - 2*XY
    ans=ans/(np.linalg.norm(ans))
    return ans


# In[20]:


result=[]
for t in range(1,1000):
    W=W-(1/t)*derivative(W)
    result.append(W)


# In[21]:


def err(w):
    return np.linalg.norm(w-Wml)


# In[22]:


error = [err(w) for w in result]


# In[16]:


plt.xlabel("Number of Iterations")
plt.ylabel("Error")
plt.plot(error)
plt.show()


# In[25]:


print(np.linalg.norm(np.matmul(XT,result[-1]) - Y)**2)


# In[26]:


print(np.linalg.norm(np.matmul(Xtest,result[-1]) - Ytest)**2)


# In[ ]:




