#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random


# In[3]:


df=pd.read_csv('A2Q1.csv',header=None)
data = df.to_numpy()


# In[4]:


def computeTheta(lamda,k):
    sum1 = 0
    for i in range(400):
        sum1 += lamda[i][k]*data[i]
    s=0
    for i in range(400):
        s= s + lamda[i][k]
    return sum1/s


# In[8]:


def computePie(lamda,k):
    s=0
    for i in range(400):
        s+= lamda[i][k]
    ans=s/400
    return ans


# In[9]:


def probabilityMassFunction(i,k,theta):
    temp=0
    res1=0
    res2=0
    product = 1
    for j in range(50):
        res1=((theta[k][j])**data[i][j])
        res2=((1 - theta[k][j])**(1 - data[i][j]))
        temp = res1*res2
        product *= temp
    return product


# In[10]:


def computeLamda(i,k,pie,theta):
    denom= 0
    numer= probabilityMassFunction(i,k,theta)*pie[k]
    for j in range(4):
        denom += probabilityMassFunction(i,j,theta)*pie[j]
    ans=numer/denom
    return ans


# In[11]:


def randomInitialization():
    listLamda = []
    theta= []
    pie = []
    for x in range(400):
        temp=0
        lis = []
        for k in range(4):
            res1=0
            ans=np.random.randint(0,100)
            lis.append(ans)
        s = sum(lis)
        lis = [i/s for i in lis]
        listLamda.append(lis)
    for k in range(4):
        res2=computeTheta(listLamda,k)
        theta.append(res2)
    for k in range(4):
        s = 0
        for i in range(400):
            s = s + listLamda[i][k]
        res3=s/400
        pie.append(res3)
    return listLamda,pie,theta


# In[22]:


def EM(data,theta,pie):
    lamda1 = []
    theta1= []
    pie1 = []
    tem=0
    for i in range(400):
        res1=0
        lis = []
        for k in range(4):
            res2=0
            res=computeLamda(i,k,pie,theta)
            lis.append(res)
        lamda1.append(lis)
    for k in range(4):
        theta1.append(computeTheta(lamda1,k))
    for k in range(4):
        res=computePie(lamda1,k)
        pie1.append(res)
    return theta1,pie1,lamda1


# In[23]:


def computeLoglikelihood(pie,theta):
    res=0
    for i in range(400):
        ans = 0
        for k in range(4):
            temp=probabilityMassFunction(i,k,theta)
            ans+=pie[k]*temp
        t=math.log10(ans)
        res+=t
    return res


# In[24]:


lamdas = []
thetas = []
pies = []
for i in range(100):
    lamdas.append(0)
    thetas.append(0)
    pies.append(0)
for i in range(100):
    lamdas[i],pies[i],thetas[i] = randomInitialization()


# In[ ]:


loglikelihood = []
for i in range(100):
    loglikelihood.append(0)
for i in range(100):
    res=computeLoglikelihood(pies[i],thetas[i])
    theta = thetas[i]
    pie = pies[i]
    loglikelihood[i] = [res]
    lamda = lamdas[i]
    res1=0
    for t in range(50):
        res2=0
        theta,pie,lamda = EM(data,theta,pie)
        res=computeLoglikelihood(pie,theta)
        loglikelihood[i].append(res)


# In[32]:


avgloglikelihood = []
for i in range(50):
    avgloglikelihood.append(0)
for i in range(100):
    temp=[]
    for j in range(50):
        res=0
        avgloglikelihood[j]+= loglikelihood[i][j]
avgloglikelihood = [x/100 for x in avgloglikelihood]


# In[36]:


plt.xlabel("Number of Iterations")
plt.ylabel("Average log-likelihood")
plt.plot(avgloglikelihood[1:])
plt.show()

