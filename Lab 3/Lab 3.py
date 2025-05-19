#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from fastapi.testclient import TestClient
import hyperdx


# In[2]:


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# In[3]:


# Load model
model = joblib.load("xgb_model.pkl")
app = FastAPI()


# In[4]:


# Define input schema
class CustomerData(BaseModel):
    CreditScore: int
    Gender: int
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float
    Geography_Germany: int
    Geography_Spain: int


# In[5]:


# Home endpoint
@app.get("/")
def home():
    logger.info("Home endpoint hit.")
    return {"message": "Welcome to Churn Prediction API"}


# In[6]:


# Health check
@app.get("/health")
def health():
    logger.info("Health check passed.")
    return {"status": "Healthy"}


# In[7]:


# Predict
@app.post("/predict")
def predict(data: CustomerData):
    logger.info(f"Received data for prediction: {data}")
    
    features = np.array([
        [
            data.CreditScore, data.Gender, data.Age, data.Tenure,
            data.Balance, data.NumOfProducts, data.HasCrCard,
            data.IsActiveMember, data.EstimatedSalary,
            data.Geography_Germany, data.Geography_Spain
        ]
    ])
    prediction = model.predict(features)
    logger.info(f"Prediction result: {prediction[0]}")
    return {"churn_prediction": int(prediction[0])}


# In[8]:


#uvicorn main:app --reload


# In[9]:


client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={
        "CreditScore": 600,
        "Gender": 1,
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
        "Geography_Germany": 1,
        "Geography_Spain": 0
    })
    assert response.status_code == 200
    assert "churn_prediction" in response.json()


#  # More Test Functions
# 

# In[10]:


client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Churn Prediction API"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "Healthy"}

def test_predict():
    response = client.post("/predict", json={
        "CreditScore": 600,
        "Gender": 1,
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
        "Geography_Germany": 1,
        "Geography_Spain": 0
    })
    assert response.status_code == 200
    assert "churn_prediction" in response.json()


# In[ ]:


hyperdx.configure(api_key="YOUR_HYPERDX_API_KEY") #i try it with my api key
logger = hyperdx.getLogger("churn-api")
logger.info("Server started")


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




