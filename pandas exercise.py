#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
df_house = pd.read_csv("housing.csv")


# In[2]:


df_house.head(10)


# In[3]:


df_house.shape


# In[9]:


df_house.columns


# In[4]:


df_house.mean()


# In[10]:


df_house.describe()


# In[8]:


df_house['price'].mean()


# In[9]:


df_house['price'].max()


# In[10]:


df_house['price'].min()


# In[12]:


df_house['price'].isna().sum()


# In[21]:


df_house['bedrooms'].isnull().sum()


# In[24]:


df_house['sqft_living'].describe()


# In[25]:


df_house.columns


# In[32]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('housing.csv')

# Display basic information about the dataset
print(df.info())

# Display summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())



# In[27]:


# Visualize the distribution of the target variable 'price'
plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.show()


# In[28]:


# Visualize correlations between numerical features
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[29]:


# Explore relationships between features and the target variable
# For example, visualize the relationship between 'sqft_living' and 'price'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sqft_living', y='price', data=df)
plt.title('Relationship between Sqft Living and Price')
plt.xlabel('Sqft Living Area')
plt.ylabel('Price')
plt.show()


# In[30]:


# Visualize the distribution of categorical variables, such as 'bedrooms'
plt.figure(figsize=(8, 6))
sns.countplot(x='bedrooms', data=df)
plt.title('Distribution of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Count')
plt.show()


# In[31]:


# Explore relationships using pairplots
sns.pairplot(df[['price', 'sqft_living', 'bedrooms', 'bathrooms']])
plt.show()


# In[ ]:




