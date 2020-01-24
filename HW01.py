#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd

# Importing the dataset. I tried importing the dataset from the link, but it didn't work.
train = pd.read_csv('C:\\Users\\valer\\OneDrive\\Documentos\\Escritorio\\train.csv', engine ='python', sep = ',')
test = pd.read_csv('C:\\Users\\valer\\OneDrive\\Documentos\\Escritorio\\test.csv', engine ='python', sep = ',')

# Append test dataset to train dataset, filling the "Survived" column of test dataset with NaN
combined = train.append(test)

# Q1: Which features are available in the dataset?
print(train.columns.values)


# In[10]:


# Q5: Which features contain blank, null or empty values?
# Q6: What are the data types for various features?

print(combined.info())


# Q5: Age, Cabin, Embarked have empty values for the train dataset. Age, Cabin and Fare have empty values for the test dataset
# 
# Q6: PassengerId – Integer, Survived – Integer, Pclass - Integer, Name – String, Sex – String, Age – Integer and Float, SibSp – Integer, Parch – Integer, Ticket – Integer and String, Fare – Float and Integer, Cabin – String, Embarked – String 
# 

# In[15]:


# Q7: To understand what is the distribution of numerical feature values across the samples, 
# please list the properties (count, mean, std, min, 25% percentile, 50% percentile, 75% percentile,
# max) of numerical features?

pd.to_numeric(combined.Age)
pd.to_numeric(combined.SibSp)
pd.to_numeric(combined.Parch)
pd.to_numeric(combined.Fare)
combined["Sex"] = combined["Sex"].astype('category')
combined["Embarked"] = combined["Embarked"].astype('category')
combined["Pclass"] = combined["Pclass"].astype('category')
combined["Survived"] = combined["Survived"].astype('category')

combined.describe()


# In[20]:


# Q8: To understand what the distribution of categorical features is, we define:  count 
# is the total number of categorical values per column; unique is the total number of unique 
# categorical values per column; top is the most frequent categorical value; freq is the total
# number of the most frequent categorical value. Please the properties (count, unique, top, freq) 
# of categorical features?

combined.astype('object').describe()
##CATEGORICAL: Sex, Embarked, Pclass, Survived


# In[ ]:




