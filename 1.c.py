#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random


# In[38]:


df=pd.read_csv('A2Q1.csv',header=None)
Xt=np.array(df)
X=Xt.T


# In[39]:


#Xt[:,0]=Xt[:,0]-f1m
#Xt[:,1]=Xt[:,1]-f2m
#X=Xt.T


# In[40]:


def initialize():  # Initializing the cluster
    Z=[]
    for i in range(400):
        Z.append(random.randint(0,3))
    return np.array(Z)


# In[41]:


def calculateMean(X,Z,k):   # Calculating the mean for each cluster
    mean_arr=[]
    for i in range(k):
        temp_arr=[]
        for j in range(400):
            if(Z[j]==i):
                temp_arr.append(X[:,j])
        mean_arr.append(np.mean(temp_arr,axis=0))
    return np.array(mean_arr)
                


# In[42]:


def recalculateCluster(X,Z,mean_arr,k): # Recalculating the mean after reassignment step
    flag=0
    for i in range(400):
        distance=[]
        for j in range(k):
            distance.append(calculateDistance(X[:,i],mean_arr[j]))
        index=np.argmin(distance)
        if(index!=Z[i]):
            flag=1
        Z[i]=index
    return np.array(Z),flag


# In[ ]:





# In[43]:


def calculateDistance(x,y):  # calculating euclidian distance
    return np.sqrt(sum(np.square(x-y)))


# In[44]:


def kmeans(k,X):  # Running LLoyd's algorithm
    flag=1
    clusterIndicator =initialize()
    err=[]
    while(flag):
        mean=calculateMean(X,clusterIndicator,k)
        newcluster=[]
        newcluster,flag=recalculateCluster(X,clusterIndicator,mean,k)
        summation =0
        for y in range(400):
            summation+=calculateDistance(X[:,y],mean[clusterIndicator[y]])
        err.append(summation)
        if(flag==0):
           break
    return err,newcluster


# In[45]:


error1,newclust1=kmeans(4,X)


# In[46]:


plt.figure()
plt.xlabel("No of Iterations")
plt.ylabel("Error")
plt.plot([i for i in range(len(error1))],error1)
plt.show()

