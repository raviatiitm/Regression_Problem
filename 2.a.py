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


XT=traindata[:,0:100]


# In[6]:


X=XT.T


# In[7]:


Y=traindata[:,100].T


# In[8]:


Xtest=testdata[:,0:100]
Ytest=testdata[:,100]


# In[9]:


XY=np.matmul(X,Y)


# In[10]:


XXT=np.matmul(X,XT)


# In[11]:


Wml=np.matmul(np.linalg.inv(XXT),XY)


# In[12]:


print(np.linalg.norm(np.matmul(XT,Wml) - Y)**2)


# In[13]:


print(np.linalg.norm(np.matmul(Xtest,Wml) - Ytest)**2)

