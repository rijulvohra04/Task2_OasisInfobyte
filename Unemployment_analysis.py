#!/usr/bin/env python
# coding: utf-8

# # TASK: Unemployment Analysis with Python

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
     


# In[4]:


data = pd.read_csv(r'C:\Users\Rijul\Downloads\Unemployment_Rate_upto_11_2020.csv')
    


# In[5]:


data.head()


# In[6]:


data.columns= ["States","Date","Frequency",
               "Estimated Unemployment Rate",
               "Estimated Employed",
               "Estimated Labour Participation Rate",
               "Region","longitude","latitude"]
     


# In[7]:


data.head()


# In[8]:


data.describe()


# In[9]:


print(data.isnull().sum())


# In[12]:


my_data=data.select_dtypes(exclude=[object])


# In[13]:


my_data.corr()


# In[15]:


plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(12, 10))
sns.heatmap(my_data.corr())
plt.show()


# ## Visualize the data

# In[16]:


data.columns= ["States","Date","Frequency",
               "Estimated Unemployment Rate","Estimated Employed",
               "Estimated Labour Participation Rate","Region",
               "longitude","latitude"]
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Employed", hue="Region", data=data)
plt.show()


# In[17]:


plt.figure(figsize=(10, 8))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate", hue="Region", data=data)
plt.show()
     


# In[ ]:


unemploment = data[["States", "Region", "Estimated Unemployment Rate"]]
figure = px.sunburst(unemploment, path=["Region", "States"], 
                     values="Estimated Unemployment Rate", 
                     width=700, height=700, color_continuous_scale="RdY1Gn", 
                     title="Unemployment Rate in India")
figure.show()


# In[19]:


sns.pairplot(data)


# In[20]:


data.describe()


# In[21]:


X = data[['Estimated Unemployment Rate', 'Estimated Employed', 'Estimated Labour Participation Rate',
       'longitude', 'latitude']]

y = data['Estimated Employed']
     


# In[22]:


from sklearn.model_selection import train_test_split
     


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40)
     


# In[24]:


X_train


# In[25]:


from sklearn.linear_model import LinearRegression
     


# In[26]:


lm = LinearRegression()


# In[28]:


lm.fit(X_train, y_train)
     


# ## Evaluating the Model

# In[29]:


coeff_data = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
     


# In[30]:


coeff_data


# ## Predicting the Model

# In[31]:


predictions = lm.predict(X_test)


# In[32]:


plt.scatter(y_test, predictions)
     


# In[33]:


sns.distplot((y_test-predictions), bins=50);


# In[ ]:



