#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
insurance=pd.read_csv("insurance.csv")


# In[4]:


insurance.head()


# In[5]:


insurance.isnull().sum()


# In[6]:


insurance.shape


# In[7]:


# getting some information about the dataset
insurance.info()


# In[8]:


#Features: *Sex,*Smoker,*Region
insurance.isnull().sum()


# In[9]:


# Data Analysis
insurance.describe()


# In[10]:


#Distribution of age value
sns.set()
#plt.figure(figsize=6,6)
sns.distplot(insurance['age'])
plt.title('Age distribution')
plt.show()


# In[11]:


# Gender Colum
sns.set()
#plt.figure(figsize=6,6)
sns.countplot(x='sex' , data=insurance)
plt.title('Sex distribution')
plt.show()


# In[12]:


insurance['sex'].value_counts()


# In[13]:


#Bmi Distribution
sns.set()
#plt.figure(figsize=6,6)
sns.distplot(insurance['bmi'])
plt.title('bmi distribution')
plt.show()


# In[14]:


#Children colum
sns.set()
#plt.figure(figsize=6,6)
sns.countplot(insurance['children'])
plt.title('children')
plt.show()


# In[15]:


insurance['children'].value_counts()


# In[16]:


#smoker colum
sns.set()
sns.countplot(insurance['smoker'])
plt.title('smoker')
plt.show()


# In[17]:


insurance['smoker'].value_counts()


# In[18]:


#region colum
sns.set()
sns.countplot(insurance['region'])
plt.title('region')
plt.show()


# In[19]:


insurance['region'].value_counts()


# In[20]:


#Charge Distribution
sns.set()
#plt.figure(figsize=6,6)
sns.distplot(insurance['charges'])
plt.title('charges distribution')
plt.show()


# In[21]:


# DATA PRE-PROCESSING
# Encoding the categorical feature

# encoding sex column
insurance.replace({'sex':{'male':0, 'female':1}}, inplace=True)

# encoding smoker column
insurance.replace({'smoker':{'yes':0, 'no':1}}, inplace=True)

#encoding region column
insurance.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}}, inplace=True)


# In[23]:


# Spliting the feature and Target

X=insurance.drop(columns='charges',axis=1)
Y=insurance['charges']

print(X)


# In[24]:


print(Y)


# In[28]:


# Spliting data into training data & testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)


# In[29]:


# Model Trainig
# Linear Regression

regressor= LinearRegression()
regressor.fit(X_train, Y_train)


# In[30]:


# Model Evaluation

# Pridiction on training data
training_data_prediction=regressor.predict(X_train)

# R squared value(lies b/t 0 to 1)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)


# In[31]:


# prediction on test data
test_data_prediction = regressor.predict(X_test)


# In[32]:


# R squared value(lies b/t 0 to 1)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)


# In[33]:


# Building a Predictive System

input_data = (31,1,25.74,0,1,0)

# chaging input_data to a numpy array
input_data_as_numpy_array = np.asanyarray(input_data)

#reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)

print('The insurance cost in USD', prediction[0])


# # The insurance cost in USD 3760.0805764960496
