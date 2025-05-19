#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install pandas 
#!pip install pandas sklearn
#!pip install -r requirements.txt  
#!pip install scikit-learn
#!pip install mlflow
#!pip install dataloader
#!pip install data-loader


# In[ ]:


#!pip install scikit-learn pandas mlflow xgboost


# In[18]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import mlflow
import mlflow.sklearn       
from mlflow.tracking import MlflowClient    
from mlflow.exceptions import MlflowException
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib


# In[2]:


mlflow.set_tracking_uri("http://localhost:5000")


# In[ ]:


#run this in another terminal to see the UI
#python -m mlflow ui


# In[3]:


exp_id=mlflow.create_experiment(name="bank churn")
exp_id


# In[4]:


data = pd.read_csv("Churn_Modelling.csv")


# In[5]:


data.head()


# In[6]:


data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
data = pd.get_dummies(data, columns=['Geography'], drop_first=True)


# In[7]:


# Features and target
X = data.drop('Exited', axis=1)
y = data['Exited']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# In[9]:


# Predictions
y_pred = model.predict(X_test)


# In[10]:


accuracy = accuracy_score(y_test, y_pred)


# In[11]:


with mlflow.start_run():
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")


# In[12]:



rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_preds)

with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", rf_accuracy)
    mlflow.sklearn.log_model(rf_model, "model")


# In[13]:


xgb_model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_preds)

with mlflow.start_run():
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", xgb_accuracy)
    mlflow.sklearn.log_model(xgb_model, "model")


# In[14]:


with mlflow.start_run():
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", rf_accuracy)
    
    # Register the model
    mlflow.sklearn.log_model(
        rf_model,
        artifact_path="model",
        registered_model_name="RandomForestClassifier"
    )


# In[15]:


with mlflow.start_run():
    mlflow.log_param("model", "XGBoost")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", xgb_accuracy)
    
    mlflow.sklearn.log_model(
        xgb_model,
        artifact_path="model",
        registered_model_name="XGBoostClassifier"
    )


# In[17]:


model_scores = {}
model_scores["Logistic Regression"] = accuracy
model_scores["Random Forest"] = rf_accuracy      
model_scores["XGBoost"] = xgb_accuracy  
print("\nModel Accuracy Comparison:")
for model, acc in model_scores.items():
    print(f"{model}: {acc:.4f}")

best_model = max(model_scores, key=model_scores.get)
print(f"\nBest Model: {best_model} with accuracy {model_scores[best_model]:.4f}")


# In[19]:


# Save model after training
joblib.dump(xgb_model, "xgb_model.pkl")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




