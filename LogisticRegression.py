#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# In[4]:


data = pd.read_csv('diabetes.csv')
data.tail()


# Clean the data

# In[5]:


clean_data = ['Glucose','BloodPressure','SkinThickness','BMI','Insulin']

for col in clean_data:
  data[col]= data[col].replace(0,np.NaN)
  mean = int(data[col].mean(skipna = True))
  data[col] = data[col].replace(np.NaN,mean)


# split data

# In[6]:


x = data.drop(labels = "Outcome", axis = 1)
y = data["Outcome"]


# In[7]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[8]:


standard_scaler = StandardScaler()
scaled_X_train = pd.DataFrame(standard_scaler.fit_transform(x_train), index = x_train.index, columns = x_train.columns)
scaled_X_test = pd.DataFrame(standard_scaler.fit_transform(x_test), index = x_test.index, columns = x_test.columns)


# In[9]:


lr = LogisticRegression()
lr.fit(scaled_X_train,y_train)
y_predict = lr.predict(scaled_X_test)

print(y_predict)

score = accuracy_score(y_test,y_predict)


# In[10]:


print(score*100)


# Confusion Matrix

# In[11]:


from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test,y_predict)
cnf_matrix


# We have two classes 0 and 1. Diagonal values represent accurate predictions, while non-diagonal elements are inaccurate predictions (97 & 29 accurate) else inaccurate.

# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# Text(0.5,257.44,'Predicted label');


# In[13]:


from sklearn.metrics import classification_report
target_names = ['without diabetes', 'with diabetes']
print(classification_report(y_test, y_predict, target_names=target_names))


# ROC Curve

# In[14]:


y_pred_proba = lr.predict_proba(scaled_X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()



# In[16]:


import joblib

# Assuming 'lr' is your trained LogisticRegression model
joblib.dump(lr, 'logistic_regression_model.pkl')


# In[17]:


# import joblib

# # Load the saved model
# loaded_lr = joblib.load('logistic_regression_model.pkl')


# In[ ]:




