#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import warnings
warnings.filterwarnings('ignore')


# In[22]:


import pandas as pd
url = 'https://raw.githubusercontent.com/dsrscientist/dataset1/master/mushrooms.csv'
df = pd.read_csv(url, error_bad_lines=False)


# In[33]:


print (df)


# In[28]:


df.shape


# In[35]:


df.info()


# In[36]:


df.isnull().sum()


# In[37]:


df.dtypes


# In[38]:


sns.heatmap(df.isnull())


# In[39]:


x=df.drop('class',axis=1) #Predictors
y=df['class'] #target
x.head()


# In[40]:


y.head()


# In[41]:


from sklearn.preprocessing import LabelEncoder
Encoder_X = LabelEncoder() 
for col in x.columns:
    x[col] = Encoder_X.fit_transform(x[col])
Encoder_y=LabelEncoder()
y = Encoder_y.fit_transform(y)


# In[42]:


x.head()


# In[43]:


y


# In[45]:


df.describe()


# In[46]:


from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()
le = LabelEncoder()
for col in df_encoded.columns:
    df_encoded[col] = le.fit_transform(df_encoded[col]) 
    
df_encoded.head()


# In[47]:


df_encoded.describe()


# In[48]:


#poisonous=1
#eatable=0


# In[49]:


df['odor'].value_counts()


# In[50]:


df['class'].value_counts()


# In[51]:


#we are taking here [e] as no-eatables and [p] as eatables:


# In[52]:


df.columns


# # Visualisation of Data

# In[53]:


import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


# In[58]:


def plot_col(col, hue=None, color=['lightblue', 'pink'], labels=None):
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.countplot(col, hue=hue, palette=color, saturation=0.6, data=df, dodge=True, ax=ax)
    ax.set(title = f"Mushroom {col.title()} Quantity", xlabel=f"{col.title()}", ylabel="Quantity")
    if labels!=None:
        ax.set_xticklabels(labels)
    if hue!=None:
        ax.legend(('Poisonous', 'Edible'), loc=0)


# In[59]:


class_dict = ('Poisonous', 'Edible')
plot_col(col='class', labels=class_dict)


# In[60]:


shape_dict = {"bell":"b","conical":"c","convex":"x","flat":"f", "knobbed":"k","sunken":"s"}
labels = ('convex', 'bell', 'sunken', 'flat', 'knobbed', 'conical')
plot_col(col='cap-shape', hue='class', labels=labels)


# In[64]:


color_dict = {"yellow":"n","black":"y", "blue":"w", "gray":"g", "red":"e","purple":"p",
              "orange":"b", "pink":"u", "brown":"c", "green":"r"}
plot_col(col='cap-color', color=color_dict.keys(), labels=color_dict)


# In[65]:


plot_col(col='cap-color', hue='class', labels=color_dict)


# In[66]:


surface_dict = {"smooth":"s", "scaly":"y", "fibrous":"f","grooves":"g"}
plot_col(col='cap-surface', hue='class', labels=surface_dict)


# In[67]:


def get_labels(order, a_dict):    
    labels = []
    for values in order:
        for key, value in a_dict.items():
            if values == value:
                labels.append(key)
    return labels


# In[68]:


odor_dict = {"almond":"a","anise":"l","creosote":"c","fishy":"y",
             "foul":"f","musty":"m","none":"n","pungent":"p","spicy":"s"}
order = ['p', 'a', 'l', 'n', 'f', 'c', 'y', 's', 'm']
labels = get_labels(order, odor_dict)      
plot_col(col='odor', color=color_dict.keys(), labels=labels)


# # Testing and Training of Data

# In[69]:


from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x=scale.fit_transform(x)


# In[70]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=52)


# In[83]:


x_train.shape, x_test.shape


# In[85]:


y_train.shape, y_test.shape


# In[86]:


lg=LogisticRegression()
lg.fit(x_train,y_train)


# In[114]:


pred=lg.predict(x_test)
print("accuracy_score :",accuracy_score(pred,y_test))
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[115]:


model=[KNeighborsClassifier(),GaussianNB(),SVC(),DecisionTreeClassifier()]


# In[116]:


for k in model:
    k.fit(x_train,y_train)
    k.score(x_train,y_train)
    predm=k.predict(x_test)
    print('Accuracy score of',k,'is :')
    print(accuracy_score(y_test,predm))
    print(confusion_matrix(y_test,predm))
    print(classification_report(y_test,predm))
    print('\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




