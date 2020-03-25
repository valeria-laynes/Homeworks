#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import packages

from surprise import *
import os
from surprise.model_selection import *
from surprise import accuracy
from surprise import Dataset
import numpy as np
import random
import matplotlib.pyplot as plt


# In[2]:


# 3 Import dataset

np.random.seed(0)
random.seed(0)

file_path = os.path.expanduser('C:\\Users\\valer\\OneDrive\\Documentos\\Escritorio\\restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)
#trainset, testset = model_selection.split.train_test_split(data, test_size=0.2, train_size=None, random_state=None, shuffle=False)
#kf = model_selection.split.KFold(n_splits = 3, shuffle = False)
SS = model_selection.split.ShuffleSplit(n_splits=3, test_size=0.2, train_size=None, random_state=None, shuffle=False)


# In[3]:


Fold1,Fold2,Fold3= SS.split(data)
trainset1,testset1 = Fold1
trainset2,testset2 = Fold2
trainset3,testset3 = Fold3


# In[4]:


# 5 Cross validation SVD
algo = SVD()

algo.fit(trainset1)
predictions1 = algo.test(testset1)
print('Pred 1, RSME: ',accuracy.rmse(predictions1))
print('Pred 1, MAE: ',accuracy.mae(predictions1))

algo.fit(trainset2)
predictions2 = algo.test(testset2)
print('Pred 2, RSME: ',accuracy.rmse(predictions2))
print('Pred 2, MAE: ',accuracy.mae(predictions2))

algo.fit(trainset3)
predictions3 = algo.test(testset3)
print('Pred 3, RSME: ',accuracy.rmse(predictions3))
print('Pred 3, MAE: ',accuracy.mae(predictions3))

print('Mean SVD RSME: ', (accuracy.rmse(predictions1) + accuracy.rmse(predictions2) + accuracy.rmse(predictions3))/3)
print('Mean SVD MAE: ', (accuracy.mae(predictions1) + accuracy.mae(predictions2) + accuracy.mae(predictions3))/3)


# In[5]:


# 6 Cross validation PMF

algo = SVD(biased=False)
algo.fit(trainset1)
predictions1 = algo.test(testset1)
print('Pred 1, RSME: ',accuracy.rmse(predictions1))
print('Pred 1, MAE: ',accuracy.mae(predictions1))

algo.fit(trainset2)
predictions2 = algo.test(testset2)
print('Pred 2, RSME: ',accuracy.rmse(predictions2))
print('Pred 2, MAE: ',accuracy.mae(predictions2))

algo.fit(trainset3)
predictions3 = algo.test(testset3)
print('Pred 3, RSME: ',accuracy.rmse(predictions3))
print('Pred 3, MAE: ',accuracy.mae(predictions3))

print('Mean PMF RSME: ', (accuracy.rmse(predictions1) + accuracy.rmse(predictions2) + accuracy.rmse(predictions3))/3)
print('Mean PMF MAE: ', (accuracy.mae(predictions1) + accuracy.mae(predictions2) + accuracy.mae(predictions3))/3)


# In[6]:


# 7 Cross validation NMF

algo = NMF()
algo.fit(trainset1)
predictions1 = algo.test(testset1)
print('Pred 1, RSME: ',accuracy.rmse(predictions1))
print('Pred 1, MAE: ',accuracy.mae(predictions1))

algo.fit(trainset2)
predictions2 = algo.test(testset2)
print('Pred 2, RSME: ',accuracy.rmse(predictions2))
print('Pred 2, MAE: ',accuracy.mae(predictions2))

algo.fit(trainset3)
predictions3 = algo.test(testset3)
print('Pred 3, RSME: ',accuracy.rmse(predictions3))
print('Pred 3, MAE: ',accuracy.mae(predictions3))

print('Mean NMF RSME: ', (accuracy.rmse(predictions1) + accuracy.rmse(predictions2) + accuracy.rmse(predictions3))/3)
print('Mean NMF MAE: ', (accuracy.mae(predictions1) + accuracy.mae(predictions2) + accuracy.mae(predictions3))/3)


# In[7]:


# 8 Cross validation User based Collaborative Filtering

algo = KNNBasic(sim_options = { 'user_based': True })
algo.fit(trainset1)
predictions1 = algo.test(testset1)
print('Pred 1, RSME: ',accuracy.rmse(predictions1))
print('Pred 1, MAE: ',accuracy.mae(predictions1))

algo.fit(trainset2)
predictions2 = algo.test(testset2)
print('Pred 2, RSME: ',accuracy.rmse(predictions2))
print('Pred 2, MAE: ',accuracy.mae(predictions2))

algo.fit(trainset3)
predictions3 = algo.test(testset3)
print('Pred 3, RSME: ',accuracy.rmse(predictions3))
print('Pred 3, MAE: ',accuracy.mae(predictions3))

print('Mean UCF RSME: ', (accuracy.rmse(predictions1) + accuracy.rmse(predictions2) + accuracy.rmse(predictions3))/3)
print('Mean UCF MAE: ', (accuracy.mae(predictions1) + accuracy.mae(predictions2) + accuracy.mae(predictions3))/3)


# In[8]:


# 9 Cross validation Item Based Collaborative Filtering 

algo = KNNBasic(sim_options = { 'user_based': False })

algo.fit(trainset1)
predictions1 = algo.test(testset1)
print('Pred 1, RSME: ',accuracy.rmse(predictions1))
print('Pred 1, MAE: ',accuracy.mae(predictions1))

algo.fit(trainset2)
predictions2 = algo.test(testset2)
print('Pred 2, RSME: ',accuracy.rmse(predictions2))
print('Pred 2, MAE: ',accuracy.mae(predictions2))

algo.fit(trainset3)
predictions3 = algo.test(testset3)
print('Pred 3, RSME: ',accuracy.rmse(predictions3))
print('Pred 3, MAE: ',accuracy.mae(predictions3))

print('Mean ICF RSME: ', (accuracy.rmse(predictions1) + accuracy.rmse(predictions2) + accuracy.rmse(predictions3))/3)
print('Mean ICF MAE: ', (accuracy.mae(predictions1) + accuracy.mae(predictions2) + accuracy.mae(predictions3))/3)


# In[9]:


# 14 UCF MSD 

algo = KNNBasic(sim_options = { 'name': 'MSD', 'user_based': True})

algo.fit(trainset1)
predictions1 = algo.test(testset1)
print('Pred 1, RSME: ',accuracy.rmse(predictions1))
print('Pred 1, MAE: ',accuracy.mae(predictions1))

algo.fit(trainset2)
predictions2 = algo.test(testset2)
print('Pred 2, RSME: ',accuracy.rmse(predictions2))
print('Pred 2, MAE: ',accuracy.mae(predictions2))

algo.fit(trainset3)
predictions3 = algo.test(testset3)
print('Pred 3, RSME: ',accuracy.rmse(predictions3))
print('Pred 3, MAE: ',accuracy.mae(predictions3))

meanRSME1 = (accuracy.rmse(predictions1) + accuracy.rmse(predictions2) + accuracy.rmse(predictions3))/3
print('Mean SVD RSME: ', meanRSME1)

meanMAE1 = (accuracy.mae(predictions1) + accuracy.mae(predictions2) + accuracy.mae(predictions3))/3
print('Mean SVD MAE: ', meanMAE1)


# In[10]:


# 14 UCF Cosine

algo = KNNBasic(sim_options = { 'name':'cosine', 'user_based': True}) 

algo.fit(trainset1)
predictions1 = algo.test(testset1)
print('Pred 1, RSME: ',accuracy.rmse(predictions1))
print('Pred 1, MAE: ',accuracy.mae(predictions1))

algo.fit(trainset2)
predictions2 = algo.test(testset2)
print('Pred 2, RSME: ',accuracy.rmse(predictions2))
print('Pred 2, MAE: ',accuracy.mae(predictions2))

algo.fit(trainset3)
predictions3 = algo.test(testset3)
print('Pred 3, RSME: ',accuracy.rmse(predictions3))
print('Pred 3, MAE: ',accuracy.mae(predictions3))

meanRSME2 = (accuracy.rmse(predictions1) + accuracy.rmse(predictions2) + accuracy.rmse(predictions3))/3
print('Mean SVD RSME: ', meanRSME2)

meanMAE2 = (accuracy.mae(predictions1) + accuracy.mae(predictions2) + accuracy.mae(predictions3))/3
print('Mean SVD MAE: ', meanMAE2)


# In[11]:


# 14 UCF Pearson

algo = KNNBasic(sim_options = { 'name':'pearson', 'user_based': True})
                               
algo.fit(trainset1)
predictions1 = algo.test(testset1)
print('Pred 1, RSME: ',accuracy.rmse(predictions1))
print('Pred 1, MAE: ',accuracy.mae(predictions1))

algo.fit(trainset2)
predictions2 = algo.test(testset2)
print('Pred 2, RSME: ',accuracy.rmse(predictions2))
print('Pred 2, MAE: ',accuracy.mae(predictions2))

algo.fit(trainset3)
predictions3 = algo.test(testset3)
print('Pred 3, RSME: ',accuracy.rmse(predictions3))
print('Pred 3, MAE: ',accuracy.mae(predictions3))

meanRSME3 = (accuracy.rmse(predictions1) + accuracy.rmse(predictions2) + accuracy.rmse(predictions3))/3
print('Mean SVD RSME: ', meanRSME3)

meanMAE3 = (accuracy.mae(predictions1) + accuracy.mae(predictions2) + accuracy.mae(predictions3))/3
print('Mean SVD MAE: ', meanMAE3)


# In[12]:


# 14 ICF MSD

algo = KNNBasic(sim_options = { 'name':'MSD', 'user_based': False})
                               
algo.fit(trainset1)
predictions1 = algo.test(testset1)
print('Pred 1, RSME: ',accuracy.rmse(predictions1))
print('Pred 1, MAE: ',accuracy.mae(predictions1))

algo.fit(trainset2)
predictions2 = algo.test(testset2)
print('Pred 2, RSME: ',accuracy.rmse(predictions2))
print('Pred 2, MAE: ',accuracy.mae(predictions2))

algo.fit(trainset3)
predictions3 = algo.test(testset3)
print('Pred 3, RSME: ',accuracy.rmse(predictions3))
print('Pred 3, MAE: ',accuracy.mae(predictions3))

meanRSME4 = (accuracy.rmse(predictions1) + accuracy.rmse(predictions2) + accuracy.rmse(predictions3))/3
print('Mean SVD RSME: ', meanRSME4)

meanMAE4 = (accuracy.mae(predictions1) + accuracy.mae(predictions2) + accuracy.mae(predictions3))/3
print('Mean SVD MAE: ', meanMAE4)


# In[13]:


# 14 ICF Cosine

algo = KNNBasic(sim_options = { 'name':'cosine', 'user_based': False}) 
                               
algo.fit(trainset1)
predictions1 = algo.test(testset1)
print('Pred 1, RSME: ',accuracy.rmse(predictions1))
print('Pred 1, MAE: ',accuracy.mae(predictions1))

algo.fit(trainset2)
predictions2 = algo.test(testset2)
print('Pred 2, RSME: ',accuracy.rmse(predictions2))
print('Pred 2, MAE: ',accuracy.mae(predictions2))

algo.fit(trainset3)
predictions3 = algo.test(testset3)
print('Pred 3, RSME: ',accuracy.rmse(predictions3))
print('Pred 3, MAE: ',accuracy.mae(predictions3))

meanRSME5 = (accuracy.rmse(predictions1) + accuracy.rmse(predictions2) + accuracy.rmse(predictions3))/3
print('Mean SVD RSME: ', meanRSME5)

meanMAE5 = (accuracy.mae(predictions1) + accuracy.mae(predictions2) + accuracy.mae(predictions3))/3
print('Mean SVD MAE: ', meanMAE5)


# In[14]:


# 14 ICF Pearson

algo = KNNBasic(sim_options = { 'name':'pearson', 'user_based': False})
                               
algo.fit(trainset1)
predictions1 = algo.test(testset1)
print('Pred 1, RSME: ',accuracy.rmse(predictions1))
print('Pred 1, MAE: ',accuracy.mae(predictions1))

algo.fit(trainset2)
predictions2 = algo.test(testset2)
print('Pred 2, RSME: ',accuracy.rmse(predictions2))
print('Pred 2, MAE: ',accuracy.mae(predictions2))

algo.fit(trainset3)
predictions3 = algo.test(testset3)
print('Pred 3, RSME: ',accuracy.rmse(predictions3))
print('Pred 3, MAE: ',accuracy.mae(predictions3))

meanRSME6 = (accuracy.rmse(predictions1) + accuracy.rmse(predictions2) + accuracy.rmse(predictions3))/3
print('Mean SVD RSME: ', meanRSME6)

meanMAE6 = (accuracy.mae(predictions1) + accuracy.mae(predictions2) + accuracy.mae(predictions3))/3
print('Mean SVD MAE: ', meanMAE6)


# In[15]:


# 14 PLOT

# USER BASED COLLABORAIVE FILTERING
RMSE = [meanRSME1, meanRSME2,meanRSME3]
plt.bar( ['MSD', 'Cosine', 'Pearson'], RMSE)
plt.axis([-0.5,2.5,0.6,1.1])
plt.show()

MAE = [meanMAE1, meanMAE2,meanMAE3]
plt.bar(['MSD', 'Cosine', 'Pearson'], MAE)
plt.axis([-0.5,2.5,0.6,1.1])
plt.show()

# ITEM BASED COLLABORAIVE FILTERING
RMSE2 = [meanRSME4, meanRSME5,meanRSME6]
plt.bar( ['MSD', 'Cosine', 'Pearson'], RMSE2)
plt.axis([-0.5,2.5,0.6,1.1])
plt.show()

MAE2 = [meanMAE4, meanMAE5,meanMAE6]
plt.bar(['MSD', 'Cosine', 'Pearson'], MAE2)
plt.axis([-0.5,2.5,0.6,1.1])
plt.show()


# In[16]:


# 15 best K for User Based Collaborative Filtering

kB=0
meanRSME1 = []

while (kB<=20):
    algo = KNNBasic(k=kB, sim_options = {'name':'MSD', 'user_based': True })
    algo.fit(trainset1)
    predictions1 = algo.test(testset1)

    algo.fit(trainset2)
    predictions2 = algo.test(testset2)

    algo.fit(trainset3)
    predictions3 = algo.test(testset3)

    meanRSME1.append( (accuracy.rmse(predictions1) + accuracy.rmse(predictions2) + accuracy.rmse(predictions3))/3 )
    
    kB += 1


# In[17]:


# 15 best K for User Based Collaborative Filtering PLOT

plt.bar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], meanRSME1)
plt.axis([-0.5,22,0.6,1.4])


# In[18]:


# 15 best K for Item Based Collaborative Filtering



kB=0
meanRSME2 = []

while (kB<=20):
    algo = KNNBasic(k=kB, sim_options = {'name':'MSD', 'user_based': False })
    algo.fit(trainset1)
    predictions1 = algo.test(testset1)

    algo.fit(trainset2)
    predictions2 = algo.test(testset2)

    algo.fit(trainset3)
    predictions3 = algo.test(testset3)

    meanRSME2.append( (accuracy.rmse(predictions1) + accuracy.rmse(predictions2) + accuracy.rmse(predictions3))/3 )
    
    kB += 1


# In[19]:


# 15 best K for Item Based Collaborative Filtering PLOT

plt.bar([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21], meanRSME2)
plt.axis([-0.5,22,0.6,1.5])


# In[ ]:




