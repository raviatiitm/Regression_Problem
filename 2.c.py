#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random


# In[3]:


df=pd.read_csv('A2Q2Data_train.csv',header=None)
df1=pd.read_csv('A2Q2Data_test.csv',header=None)


# In[4]:


traindata=np.array(df)
testdata=np.array(df1)


# In[5]:


W=np.zeros((100))


# In[6]:


XT=traindata[:,0:100]


# In[7]:


X=XT.T


# In[8]:


Y=traindata[:,100].T


# In[9]:


Xtest=testdata[:,0:100]
Ytest=testdata[:,100]


# In[10]:


XXT=np.matmul(X,XT)


# In[11]:


XY=np.matmul(X,Y)


# In[12]:


Wml=np.matmul(np.linalg.inv(XXT),XY)


# In[13]:


def derivative(w,xxt,xy):
    ans=2*(np.matmul(xxt,w)) - 2*xy
    ans=ans/(np.linalg.norm(ans))
    return ans


# In[14]:


result=[]
wSampled=np.zeros((100))
for i in range(1,20000):
    sampledData=df.sample(100)
    sampledData = sampledData.to_numpy()
    xtSampled=sampledData[:,0:100]
    xSampled=xtSampled.T
    ySampled=sampledData[:,100]
    xxtSampled=np.matmul(xSampled,xtSampled)
    xySampled=np.matmul(xSampled,ySampled)
    wSampled=wSampled-(1/i)*derivative(wSampled,xxtSampled,xySampled)
    result.append(wSampled)


# In[15]:


def err(w):
    return np.linalg.norm(w-Wml)


# In[16]:


error = [err(w) for w in result]


# In[17]:


plt.xlabel("Number of Iterations")
plt.ylabel("Error")
plt.plot(error)
plt.show()


# In[22]:


error[-1]


# In[23]:


Wstochastic=sum(result)/len(result)


# In[24]:


print(np.linalg.norm(np.matmul(XT,Wstochastic)-Y)**2)


# In[26]:


print(np.linalg.norm(np.matmul(Xtest,Wstochastic)-Ytest)**2)

