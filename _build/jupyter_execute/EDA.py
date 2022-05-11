#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# In[1]:


import pandas as pd
import dataframe_image as dfi
from tools import functions as f
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('data/heart_2020_cleaned.csv')
df


# In[3]:


df['Age'] = df['AgeCategory'].apply(lambda x : f.encode_age_category(x))
df


# ## Statistics of Numerical Variables

# In[4]:


numerical_summary = f.make_numerical_tbl(df)
dfi.export(numerical_summary,"figures/numerical_summary.png")
numerical_summary


# In[5]:


f.plot_frequency(df)


# ## Statistics of Categorical Variables

# In[6]:


f.plot_piechart(df)


# In[7]:


f.plot_bar(df)


# ## Reference:
# https://www.kaggle.com/code/mushfirat/heartdisease-eda-prediction
