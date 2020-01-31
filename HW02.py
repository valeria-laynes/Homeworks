#!/usr/bin/env python
# coding: utf-8

# In[99]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.neighbors as skl
import numpy as np

# Importing the dataset. I tried importing the dataset from the link, but it didn't work.
train = pd.read_csv('C:\\Users\\valer\\OneDrive\\Documentos\\Escritorio\\train.csv', engine ='python', sep = ',')
test = pd.read_csv('C:\\Users\\valer\\OneDrive\\Documentos\\Escritorio\\test.csv', engine ='python', sep = ',')

# Append test dataset to train dataset, filling the "Survived" column of test dataset with NaN
combined = train.append(test, sort = False)


# In[100]:


# Q9
combined.groupby('Pclass')['Survived'].mean()


# In[101]:


# Q10
combined.groupby('Sex')['Survived'].mean()


# In[102]:


# Q11
q11 = sns.FacetGrid(combined, col = 'Survived', legend_out = True)
q11.map(plt.hist, "Age", bins = 20)


# In[103]:


# Q12
q12 = sns.FacetGrid(combined, col = 'Survived', row = 'Pclass')
q12.map(plt.hist, 'Age', bins = 20)


# In[104]:


# Q13
q13 = sns.FacetGrid(combined, col = 'Survived', row = 'Embarked')
q13.map(sns.barplot, 'Sex', 'Fare', ci = None)


# In[105]:


# Q14
print(combined['Ticket'].describe())
print(combined.groupby('Ticket')['Survived'].mean())


# In[106]:


# Q15
print(combined.info())
print(combined['Cabin'].describe())


# In[107]:


# Q16
combined['Gender'] = combined['Sex'].map({'female':1, 'male':0}).astype(int)
train['Gender'] = train['Sex'].map({'female':1, 'male':0}).astype(int)
print(combined)


# In[114]:


# Q17
combined[['Survived','Pclass','Gender','SibSp','Parch','Fare']] = combined[['Survived','Pclass','Gender','SibSp','Parch','Fare']].fillna(0)
x_test = combined.loc[combined['Age'].isnull()]
x_test = x_test[['Survived','Pclass','Gender','SibSp','Parch','Fare', 'Age']]
x_test[['Age']] = x_test[['Age']].fillna(0)

x_train = combined.dropna(subset = ['Age'])
x_train = x_train[['Survived','Pclass','Gender','SibSp','Parch','Fare', 'Age']]

x = skl.KNeighborsRegressor(n_neighbors = 5, metric = 'euclidean')
x.fit(x_train, x_train['Age'])
x_test['Age'] = x.predict(x_test)
print(x_test)


# In[3]:


# Q18
print(combined['Embarked'].describe())
combined['Embarked'] = combined['Embarked'].fillna('S')
print(combined['Embarked'].describe())


# In[17]:


# Q19
print(combined['Fare'].mode())
combined['Fare'] = combined['Fare'].fillna(8.05)
print(combined['Fare'].describe())


# In[118]:


# Q20
train['Fare'] = pd.cut(x=train['Fare'], bins = [-0.001,7.91,14.454,31.0,512.329], right = True, labels = (0,1,2,3))
print(train['Fare'].describe())
print(train.groupby('Fare')['Survived'].mean())


# In[ ]:




