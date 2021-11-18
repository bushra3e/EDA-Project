#!/usr/bin/env python
# coding: utf-8

# ## E-commerce Exploratory Data Analysis
# _______________________

# ## Import libraries
# ___________________

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import math 
import random
import sys
import pickle
from sklearn import datasets


# ## Load Data
# ____________

# In[2]:


eco = pd.read_csv('ecommerce.csv')


# In[3]:


eco.head()


# In[4]:


eco.columns


# In[ ]:





# ## Data Overview
# 

# In[5]:


eco.info()


# In[6]:


eco.dtypes


# ## Handling the data and convert from object to float

# In[7]:


eco['Sales'] = eco['Sales'].str.replace('$',"")


# In[8]:


eco=eco[eco.Sales != ("0.xf")] 


# In[9]:


eco=eco[eco.Sales != ('0.5.26') ] 


# In[10]:


eco['Sales'] = eco['Sales'].astype(float)


# In[11]:


eco=eco[eco.Discount != ('xxx') ] 


# In[12]:


eco=eco[eco.Discount != ('test') ] 


# In[13]:


eco['Discount'] = eco['Discount'].astype(float)


# In[14]:


eco['Shipping Cost'] = eco['Shipping Cost'].str.replace('$',"")


# In[15]:


eco=eco[eco ['Shipping Cost']!= ('test') ] 


# In[16]:


eco['Shipping Cost'] = eco['Shipping Cost'].astype(float)


# In[17]:


eco['Profit'] = eco['Profit'].str.replace('$',"")


# In[18]:


eco['Profit'] = eco['Profit'].astype(float)


# In[19]:


eco=eco[eco.Quantity != ("abc")] 


# In[20]:


eco['Quantity'] = eco['Quantity'].astype(float)


# In[21]:


eco['Order Date'] = pd.to_datetime(eco['Order Date'])


# In[22]:


eco['Ship Date'] = pd.to_datetime(eco['Ship Date'])


# In[23]:


eco.shape 


# In[24]:


eco.dtypes


# ## Checking for NaN

# In[25]:


eco.isnull().sum()


# In[26]:


eco['Aging'] = eco['Aging'].fillna(eco.Aging.mean()) # fill the null with the mean 


# In[27]:


eco['Quantity'] = eco['Quantity'].fillna(eco.Quantity.mean())#fill the null with the mean 


# In[28]:


eco = eco.dropna()


# In[29]:


eco.shape 


# In[30]:


eco.isnull()


# ## Checking for duplicates

# In[31]:


print(eco.duplicated()) 


# ## Check for Outliers (Boxplot)

# In[32]:


sns.boxplot(data=eco);


# In[33]:


eco['Ship Mode'].value_counts()


# In[34]:


eco=eco[eco ['Ship Mode']!= ('45788') ] 


# In[35]:


eco.describe().round(2)


#  # Correlation 

# In[36]:


eco_subset = eco.loc[:, ["Aging","Sales","Profit","Quantity","Shipping Cost"]]


# In[37]:


eco_subset_corr = pd.DataFrame(np.corrcoef(eco_subset.T))
eco_subset_corr.columns = ["Aging","Sales","Profit","Quantity","Shipping Cost"]

eco_subset_corr.index = ["Aging","Sales","Profit","Quantity","Shipping Cost"]
eco_subset_corr


# In[38]:


#Correlation between columns of data
corrheatmap=eco.corr()
sns.heatmap(corrheatmap,annot=True)
plt.show()


# In[52]:


sns.pairplot(eco)


# ## The relationship between sales and profits through visualization

# In[40]:


Sales_Norm = (eco_subset['Sales']-np.mean(eco_subset['Sales']))/np.std(eco_subset['Sales'])
Profit_Norm = (eco_subset['Profit']-np.mean(eco_subset['Profit']))/np.std(eco_subset['Profit'])


# In[41]:


#We note the logical relationship between sales and profits
plt.figure(figsize=[8,8])
plt.title("The relationship between sales and profits")
plt.scatter (x=Profit_Norm, y=Sales_Norm)
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()  


# ## visualization data

# ## What is the most used ship mode?

# In[42]:


eco['Ship Mode'].value_counts()


# In[43]:


#We note that the most used method is standard shipping
sns.catplot(x='Ship Mode',kind='count', data=eco )
plt.title('Shipping Methods')
plt.figure(figsize=(50,10))


# ## what is the Top 10 Countries based Sales?

# In[44]:


eco['Country'].value_counts()


# In[45]:


#United States got most of online shoppers followed by Australia and France

top_10 = eco['Country'].value_counts()[:10]
top_10.plot(kind='bar',figsize=(10,8))
plt.title('Top 10 Countries based Sales') 
plt.xlabel("Country")
plt.ylabel("count") 


# ## Which Products Category people are showing interests in?

# In[46]:


eco['Product'].value_counts()


# In[47]:


#The best selling products are sports wear
top = eco['Product'].value_counts()[:20]
top.plot(kind='bar',figsize=(9,9))
plt.title('Product Sold in Online Sale')
plt.xlabel("Product")
plt.ylabel("count")


# ## What is the Top 10 Spending Customer?

# In[48]:



plt.xticks(rotation = 'vertical')
plt.title('Top 10 Spending Customer');
sns.barplot(y = 'Sales', x = 'Customer ID', data = eco.head(20), alpha = .5 )


# ## The Relationship between the product in store and the sales?

# In[49]:



plt.xticks(rotation = 'vertical')
plt.title('Top 10 Spending Customer');
sns.barplot(y = 'Sales', x = 'Aging', data = eco.head(20), alpha = .5 )


# In[50]:


#fig, eco = plt.subplots(figsize=(12, 12))

# Add x-axis and y-axis
#eco.bar(eco['Ship Date'],
#       eco['Order Date'],
 #      color='purple')


# In[51]:


#eco.plot(kind='bar', stacked=True)
#plt.title("Total Pie Consumption")
#plt.xlabel("Order Date")
#plt.ylabel("Ship Date")


# In[ ]:




