#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import random


# In[16]:


weight_change_df = pd.read_csv("C:/Users/bahla/Desktop/INST414/weight_change_dataset.csv", index_col = 'Participant ID')


# In[17]:


weight_change_df_subset = weight_change_df[['Age', 'Current Weight (lbs)', 'BMR (Calories)', 'Daily Calories Consumed', 'Weight Change (lbs)','Final Weight (lbs)']]


# In[18]:


weight_change_df_subset.head(60)


# In[31]:


kmeans = KMeans(n_clusters = 6, random_state = 42)


# In[32]:


cluster_labels = kmeans.fit_predict(weight_change_df_subset)


# In[33]:


weight_change_df_subset = weight_change_df_subset.copy()


# In[34]:


weight_change_df_subset.loc[:, 'cluster'] = cluster_labels


# In[35]:


print(weight_change_df_subset[:6])


# In[36]:


grouped = weight_change_df_subset.groupby('cluster')


# In[37]:


print(grouped.get_group(5)[:6])


# In[38]:


for cluster, group in grouped:
    print(f"\ncluster{cluster}:")
    sample_participant = group.sample(n = 3).index
    for individual in sample_participant:
        print(individual)
        
        

