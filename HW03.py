#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt


# In[14]:


#TASK 4: QUESTION 1
# Import database and converting to numbers
question1 = pd.read_excel('C:\\Users\\valer\\OneDrive\\Documentos\\Escritorio\\Question1.xlsx')
question1['Is'] = question1['Is'].map({'Home':1, 'Away':0}).astype(int)
question1['Top'] = question1['Top'].map({'In':1, 'Out':0}).astype(int)
question1['Media'] = question1['Media'].map({'NBC':0, 'ABC':1, 'ESPN':2, 'FOX':3}).astype(int)
question1['Label'] = question1['Label'].fillna(0)
question1['Label'] = question1['Label'].map({'Win':1, 'Lose':0, 0:0}).astype(int)

# Partitioning data
train11 = question1[0:6]
x_train = train11[['Is', 'Top', 'Media']]
y_train = train11[['Label']]
test = question1[6:13]
x_test = test[['Is', 'Top', 'Media']]
y_test = test[['Label']]

# id3
clfid = tree.DecisionTreeClassifier(criterion='entropy')
clfid = clfid.fit(x_train, y_train)
y_test = clfid.predict(x_test)
tree.plot_tree(clfid)
plt.show()
print("Predicton of value for ID3 is: ")
print(y_test)

# C4.5 is coded in R 

# CART
clfcar = tree.DecisionTreeClassifier(criterion='gini')
clfcar = clfcar.fit(x_train, y_train)
y_test = clfcar.predict(x_test)
tree.plot_tree(clfcar)
plt.show()
print("Predicton of value for CART is: ")
print(y_test)


# In[11]:


#TASK 4: QUESTION 2
# Import database and converting to numbers
question2 = pd.read_excel('C:\\Users\\valer\\OneDrive\\Documentos\\Escritorio\\Question2.xlsx')
question2['Outlook'] = question2['Outlook'].map({'Sunny':0, 'Overcast':1, 'Rainy':2}).astype(int)
question2['Temperature'] = question2['Temperature'].map({'Hot':0, 'Mild':1, 'Cool':2}).astype(int)
question2['Humidity'] = question2['Humidity'].map({'High':0, 'Normal':1}).astype(int)
question2['Windy'] = question2['Windy'].map({'F':0, 'T':1}).astype(int)
question2['Play'] = question2['Play'].fillna(0)
question2['Play'] = question2['Play'].map({'Yes':1, 'No':0, 0:0}).astype(int)

# Partitioning data
train12 = question2[0:14]
x_train = train12[['Outlook', 'Temperature', 'Humidity','Windy']]
y_train = train12[['Play']]
test = question2[14:15]
x_test = test[['Outlook', 'Temperature', 'Humidity','Windy']]
y_test = test[['Play']]

# id3
clf2id = tree.DecisionTreeClassifier(criterion='entropy')
clf2id = clfid.fit(x_train, y_train)
y_test = clf2id.predict(x_test)
tree.plot_tree(clf2id)
plt.show()
print("Predicton of value for ID3 is: ")
print(y_test)

# C4.5 is coded in R 

# CART
clf2car = tree.DecisionTreeClassifier(criterion='gini')
clf2car = clfcar.fit(x_train, y_train)
y_test = clf2car.predict(x_test)
tree.plot_tree(clf2car)
plt.show()
print("Predicton of value for CART is: ")
print(y_test)


# In[30]:


#TASK 5: Question 1 - ID3 Model (C4.5 was built in R)
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

## Importing dataset and change to numbers
task5 = pd.read_excel('C:\\Users\\valer\\OneDrive\\Documentos\\Escritorio\\task5.xlsx')
task5['Is'] = task5['Is'].map({'Home':1, 'Away':0}).astype(int)
task5['Opponent'] = task5['Opponent'].map({'In':1, 'Out':0}).astype(int)
task5['Media'] = task5['Media'].map({'NBC':0, 'ABC':1, 'ESPN':2, 'FOX':3, 'CBS':4}).astype(int)
task5['Label'] = task5['Label'].fillna(0)
task5['Label'] = task5['Label'].map({'Win':1, 'Lose':0, 0:0}).astype(int)

## Partitioning data
xtrain = task5.loc[0:23,'Is':'Media']
ytrain = task5.loc[0:23,'Label']
xtest = task5.loc[23:36,'Is':'Media']
ytest = task5.loc[23:36,'Label']

# ID3 Decision tree
clf5 = tree.DecisionTreeClassifier(criterion='entropy')
clf5 = clf5.fit(xtrain, ytrain)
ypred = clf5.predict(xtest)
tree.plot_tree(clf5)
plt.show()
print("Predicton of values for ID3 is: ")
print(ypred)

# Performance metrics for ID3 Model
print('Accuracy: ')
print(accuracy_score(ytest, ypred))
print('Precision: ')
print(precision_score(ytest, ypred))
print('Recall: ')
print(recall_score(ytest, ypred))
print('F1: ')
print(f1_score(ytest, ypred))


# In[ ]:




