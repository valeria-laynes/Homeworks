#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[3]:


#Importing dataset
X, y = load_iris(return_X_y=True)


# In[4]:


#Cross-validation and KNN model
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# Definition of arrays
scores_k = []
scores_dt = []
arrayk = []
arraydt = []
xaxis = [1,2,3,4,5,6,7,8]

# Loop to analyze best K in KNN
for x in xaxis:
    K = x
    neigh = KNeighborsClassifier(n_neighbors=K)
    clfcar = tree.DecisionTreeClassifier(criterion='gini')
    cv = KFold(n_splits=5, random_state='none', shuffle=False)
    # Loop for the five-fold cross validation
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        neigh.fit(X_train, y_train)
        clfcar.fit(X_train, y_train)
        scores_k.append(neigh.score(X_test, y_test))
        scores_dt.append(clfcar.score(X_test, y_test))
    arrayk.append(np.mean(scores_k))
    arraydt.append(np.mean(scores_dt))

print('KNN: mean of scores for each K')
print(arrayk)
print('\n')
print('DT: mean of scores')
print(arraydt)
print('\n')

# Bar plot to determine best K for KNN
plt.bar(xaxis,arrayk)
plt.axis([0.5,8.5,0.90,0.93])


# In[6]:


#Check accuracy scores of K = 5 and DT

X, y = load_iris(return_X_y=True)

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

scores_k = []
scores_dt = []
neigh = KNeighborsClassifier(n_neighbors=5)

clfcar = tree.DecisionTreeClassifier(criterion='gini')
cv = KFold(n_splits=5, random_state='none', shuffle=False)
# Loop for the five-fold cross validation
for train_index, test_index in cv.split(X):
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    neigh.fit(X_train, y_train)
    clfcar.fit(X_train, y_train)
    scores_k.append(neigh.score(X_test, y_test))
    scores_dt.append(clfcar.score(X_test, y_test))

print(np.mean(scores_k))
print(np.mean(scores_dt))

y = np.arange(5)
plt.bar(y + 0.00, scores_k, color = 'b', width = 0.25)
plt.bar(y + 0.25, scores_dt, color = 'r', width = 0.25)
plt.axis([-0.5,5,0.65,1])


# In[ ]:




