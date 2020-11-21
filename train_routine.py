#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import sys
from sklearn.utils import resample
from model_def import *

if len(sys.argv)<4:	
	print('Missing arguements, correct format: <training_data_file_name> <text_column> <target_column>')
	sys.exit()

# In[2]:


data = pd.read_csv('data/'+sys.argv[1])


# In[3]:


data.columns


# In[4]:


data = data[[sys.argv[2],sys.argv[3]]]
data['stars'] = data[sys.argv[3]].apply(lambda x: 0 if x<=2 else 1)
print(data[sys.argv[3]].value_counts())


# In[5]:


#Re-Sample due to class imbalance
data = pd.concat([resample(data[data[sys.argv[3]]==1], replace=False, n_samples=len(data[data[sys.argv[3]]==0])),data[data[sys.argv[3]]==0]])
data = data.sample(frac=1)


# In[6]:

#myModel(dataframe, to_process_column, target_column)
model = myModel(data,sys.argv[2],sys.argv[3])


# In[7]:


model.train()


# In[10]:

#Dump model
model.dump()