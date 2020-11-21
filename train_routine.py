#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
from sklearn.utils import resample
from model_def import *


# In[2]:


data = pd.read_csv('data/yelp.csv')


# In[3]:


data.columns


# In[4]:


data = data[['text','stars']]
data['stars'] = data['stars'].apply(lambda x: 0 if x<=2 else 1)
print(data['stars'].value_counts())


# In[5]:


#Re-Sample due to class imbalance
data = pd.concat([resample(data[data['stars']==1], replace=False, n_samples=len(data[data['stars']==0])),data[data['stars']==0]])
data = data.sample(frac=1)


# In[6]:

#myModel(dataframe, to_process_column, target_column)
model = myModel(data,'text','stars')


# In[7]:


model.train()


# In[10]:

#Dump model
model.dump()