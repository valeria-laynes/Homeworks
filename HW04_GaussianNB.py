#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


# In[10]:


#Importing and partitioning dataset
dataset = pd.read_excel('C:\\Users\\valer\\OneDrive\\Documentos\\Escritorio\\hw04.xlsx')
dataset['Is'] = dataset['Is'].map({'Home':1, 'Away':0}).astype(int)
dataset['Opponent'] = dataset['Opponent'].map({'In':1, 'Out':0}).astype(int)
dataset['Media'] = dataset['Media'].map({'NBC':0, 'ABC':1, 'ESPN':2, 'FOX':3, 'CBS':4}).astype(int)
dataset['Label'] = dataset['Label'].map({'Win':1, 'Lose':0, 0:0}).astype(int)
xtrain = dataset.loc[0:23,'Is':'Media']
ytrain = dataset.loc[0:23,'Label']
xtest = dataset.loc[24:35,'Is':'Media']
ytest = dataset.loc[24:35,'Label']


# In[11]:


#NBM
gnb = GaussianNB()
ypred = gnb.fit(xtrain, ytrain).predict(xtest)
print(ypred)
print(ytest)


# In[12]:


#Performance Metrics
print('Accuracy: ')
print(accuracy_score(ytest, ypred))
print('Precision: ')
print(precision_score(ytest, ypred))
print('Recall: ')
print(recall_score(ytest, ypred))
print('F1: ')
print(f1_score(ytest, ypred))


# In[ ]:




