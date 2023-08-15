#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random


# In[27]:


df=pd.read_csv('A2Q2Data_train.csv',header=None)
df1=pd.read_csv('A2Q2Data_test.csv',header=None)


# In[53]:


data=np.array(df)
xt=data[:,0:100]
y=data[:,100]


# In[28]:


sampledData=df.sample(frac=0.8)


# In[29]:


validateData = df.drop(sampledData.index).to_numpy()


# In[30]:


Xvalidate = np.transpose(validateData[:,0:100])
Yvalidate = validateData[:,100]


# In[31]:


traindata = sampledData.to_numpy()
testdata=np.array(df1)


# In[32]:


XT=traindata[:,0:100]


# In[33]:


X=XT.T


# In[34]:


Y=traindata[:,100].T


# In[35]:


Xtest=testdata[:,0:100]
Ytest=testdata[:,100]


# In[36]:


XXT=np.matmul(X,XT)


# In[37]:


XY=np.matmul(X,Y)


# In[38]:


Wml=np.matmul(np.linalg.inv(XXT),XY)


# In[39]:


def derivative(w,lamda):
    ans=2*(np.matmul(XXT,w)) - 2*XY +2*lamda*w
    ans=ans/(np.linalg.norm(ans))
    return ans


# In[40]:


def Ridge(W,lamda):
    for t in range(1,1000):
        W=W-(1/t)*derivative(W,lamda)
    return W


# In[41]:


listW=[]
listLamda=[]
i = 0.02
for _ in range(1000):
    W=np.zeros((100))
    ans=Ridge(W,i)
    listW.append(ans)
    listLamda.append(i)
    i+=0.02


# In[42]:


def err(w):
    return np.linalg.norm(np.matmul(Xvalidate.T,w)-Yvalidate)**2


# In[43]:


error = [err(w) for w in listW]


# In[45]:


plt.xlabel("Number of Iterations")
plt.ylabel("Error with respect")
plt.plot(listLamda,error)
plt.show()


# In[47]:


minlamda=listLamda[error.index(min(error))]


# In[55]:


print(minlamda)


# In[49]:


W1=np.zeros((100))
ans=Ridge(W1,minlamda)


# In[56]:


print(np.linalg.norm(np.matmul(xt,ans)-y)**2)


# In[50]:


print(np.linalg.norm(np.matmul(Xtest,ans)-Ytest)**2)

