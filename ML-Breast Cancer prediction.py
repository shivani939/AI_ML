#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv("https://raw.githubusercontent.com/ingledarshan/AIML-B2/main/data.csv")


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.columns


# In[8]:


df.info()


# In[9]:


df['Unnamed: 32']


# In[10]:


df = df.drop("Unnamed: 32", axis=1)


# In[11]:


df.columns


# In[12]:


df.drop("id", axis=1, inplace=True) 
# df = df.drop("id", axis=1)


# In[13]:


df.columns


# In[14]:


type(df.columns)


# In[16]:


l = list(df.columns)
print(l)


# In[17]:


features_mean = l[1:11]

features_se = l[11:20]

features_worst = l[21:]


# In[18]:


print(features_mean)


# In[19]:


print(features_se)


# In[21]:


print(features_worst)


# In[22]:


df.head(2)


# In[23]:


df['diagnosis'].unique()


# In[24]:


df['diagnosis'].value_counts()


# In[25]:


df.shape
#for no of rows and columns


# In[26]:


sns.barplot(x="radius_mean",data=df)


# In[28]:


sns.countplot(df['diagnosis'], label= "count")


# exploring the data

# In[27]:


#summary of all the numeric values
df.describe()


# In[30]:


len(df.columns)


# In[32]:


#create a correlation plot
corr = df.corr()
corr


# In[33]:


corr.shape


# In[37]:


plt.figure(figsize=(14,14))
sns.heatmap(corr)


# In[38]:


df.head()


# In[39]:


df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})


# In[40]:


df.head()


# In[41]:


df['diagnosis'].unique()


# In[42]:


X = df.drop('diagnosis', axis = 1)
X.head()


# In[43]:


Y = df['diagnosis']


# In[45]:


Y.head()


# In[46]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# In[47]:


df.shape


# In[48]:


X_train.shape


# In[49]:


X_test.shape


# In[50]:


Y_train.shape


# In[51]:


Y_test.shape


# In[52]:


X_train.head(2)


# In[53]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[54]:


X_train


# In[56]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, Y_train)


# In[57]:


y_pred = lr.predict(X_test)


# In[58]:


y_pred


# In[59]:


from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,y_pred))


# In[60]:


lr_acc = accuracy_score(Y_test,y_pred)


# In[61]:


results = pd.DataFrame()
results


# In[62]:


tempResults = pd.DataFrame({'Algorithm':['Logistic Regression Method'], 'Accuracy':[lr_acc]})
results = pd.concat([results, tempResults])
results = results[['Algorithm','Accuracy']]


# In[63]:


results


# In[64]:


#Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,Y_train)


# In[65]:


y_pred = dtc.predict(X_test)
y_pred


# In[66]:


from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, y_pred))


# In[67]:


dtc_acc = accuracy_score(Y_test, y_pred)


# In[68]:


tempResults = pd.DataFrame({'Algorithm':['Decision tree classifier Method'], 'Accuracy':[dtc_acc]})
results = pd.concat([results, tempResults])
results = results[['Algorithm','Accuracy']]
results


# In[71]:


#random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,Y_train)


# In[74]:


y_pred = rfc.predict(X_test)
y_pred


# In[75]:


from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, y_pred))


# In[76]:


rfc_acc = accuracy_score(Y_test, y_pred)


# In[77]:


tempResults = pd.DataFrame({'Algorithm':['Random forest classifier Method'], 'Accuracy':[rfc_acc]})
results = pd.concat([results, tempResults])
results = results[['Algorithm','Accuracy']]
results


# In[78]:


#support vector classifier
from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,Y_train)


# In[80]:


y_pred = svc.predict(X_test)
y_pred


# In[81]:


from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, y_pred))


# In[82]:


svc_acc = accuracy_score(Y_test, y_pred)


# In[83]:


tempResults = pd.DataFrame({'Algorithm':['Support vector classifier Method'], 'Accuracy':[svc_acc]})
results = pd.concat([results, tempResults])
results = results[['Algorithm','Accuracy']]
results


# In[ ]:




