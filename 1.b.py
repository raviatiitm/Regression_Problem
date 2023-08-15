#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random


# In[4]:


df=pd.read_csv('A2Q1.csv',header=None)
data = df.to_numpy()


# In[5]:


def computeMieu(lamda,k):
    sum1 = 0
    s=0
    for i in range(400):
        sum1 += lamda[i][k]*data[i]
    for i in range(400):
       s=s+ lamda[i][k]
    return sum1/s


# In[6]:


def computeSigma(mieu,lamda,k):
    sum1 = 0
    s=0
    for i in range(400):
        res=np.outer((data[i]-mieu[k]),np.transpose(data[i]-mieu[k]))
        sum1 += lamda[i][k]*res
    for i in range(400):
        s = s + lamda[i][k]
    return sum1/s


# In[7]:


def computePie(lamda,k):
    s=0
    for i in range(400):
        s = s + lamda[i][k]
    return s/400


# In[8]:


def probabilityDistribution(i,k,pie,mieu,sigma):
    res1=np.transpose(data[i]-mieu)
    res2=np.matmul(np.linalg.inv(sigma),(data[i]-mieu))
    res3=np.matmul(res1,res2)
    exp  = math.exp(-0.5*res3)
    res4=np.linalg.det(sigma)**(0.5)
    denom = 1/(((2*math.pi)*25)*res4)
    ans=pie*denom*exp
    return ans


# In[9]:


def computeLamda(i,k,pie,mieu,sigma):
    denom= 0
    numer= probabilityDistribution(i,k,pie[k],mieu[k],sigma[k])
    for kstar in range(4):
        denom += probabilityDistribution(i,kstar,pie[kstar],mieu[kstar],sigma[kstar])
    ans=numer/denom
    return ans


# In[10]:


def randomInitialization():
    listLamda = []
    mieu = []
    sigma=[]
    pie = []
    temp=0
    for x in range(400):
        res=0
        lis = []
        for k in range(4):
            res1=0
            ans=np.random.randint(0,100)
            lis.append(ans)
        s = sum(lis)
        lis = [i/s for i in lis]
        listLamda.append(lis)
    for k in range(4):
        res2=computeMieu(listLamda,k)
        mieu.append(res2)
    for k in range(4):
        res3=computeSigma(mieu,listLamda,k)
        sigma.append(res3)
    for k in range(4):
        s = 0
        for i in range(400):
            s = s + listLamda[i][k]
        res4=s/400
        pie.append(res4)
    return listLamda,pie,mieu,sigma


# In[11]:


def EM(data,mieu,sigma,pie):
    lamda1 = []
    mieu1 = []
    pie1 = []
    sigma1= []
    temp=0
    for i in range(400):
        res1=0
        lis = []
        for k in range(4):
            res2=0
            res=computeLamda(i,k,pie,mieu,sigma)
            lis.append(res)
        lamda1.append(lis)
    for k in range(4):
        res=computeMieu(lamda,k)
        mieu1.append(res)
    for k in range(4):
        res=computePie(lamda,k)
        pie1.append(res)
    for k in range(4):
        res=computeSigma(mieu,lamda,k)
        sigma1.append(res)
    return sigma1,mieu1,pie1,lamda1


# In[12]:


def computeLoglikelihood(pie,mieu,sigma):
    res= 0
    temp=0
    for i in range(400):
        lis=[]
        ans = 0
        for k in range(4):
            ans+=probabilityDistribution(i,k,pie[k],mieu[k],sigma[k])
        res+= math.log10(ans)
    return res


# In[13]:


lamdas = []
mieus= []
sigmas = []
pies = []
for i in range(100):
    lamdas.append(0)
    mieus.append(0)
    sigmas.append(0)
    pies.append(0)
for i in range(100):
    lamdas[i],pies[i],mieus[i],sigmas[i] = randomInitialization()


# In[ ]:


loglikelihood = []
for i in range(20):
    try:
        temp =[]
        mieu = mieus[i]
        sigma = sigmas[i]
        pie = pies[i]
        lamda = lamdas[i]
        val=0
        for t in range(10):
            count=[]
            sigma,mieu,pie,lamda = EM(data,mieu,sigma,pie)
            res=computeLoglikelihood(pie,mieu,sigma)
            temp.append(res)
    except:
        pass
    if(len(temp)==10):
        loglikelihood[i].append(temp)
    


# In[ ]:


avgloglikelihood = []
n=len(loglikelihood)
for i in range(n):
    avgloglikelihood.append(0)
for i in range(n):
    for j in range(10):
        avgloglikelihood[j] += loglikelihood[i][j]
avgloglikelihood = [p/n for p in avgloglikelihood]


# In[ ]:


plt.xlabel("Number of Iterations")
plt.ylabel("Average log-likelihood")
plt.plot(avgloglikelihood[1:])
plt.show()

