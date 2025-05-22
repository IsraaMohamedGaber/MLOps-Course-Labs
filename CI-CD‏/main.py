from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import logging
from prometheus_fastapi_instrumentator import Instrumentator

#import hyperdx

# Setup HyperDX (use your actual API key)
#hyperdx.configure(api_key="436872d4-dff6-4270-b4ca-3198cba74784")
#logger = hyperdx.getLogger("churn-api")
#logger.setLevel(logging.INFO)

# Initialize FastAPI app
app = FastAPI()

Instrumentator().instrument(app).expose(app)

# Load model
model = joblib.load("xgb_model.pkl")
#logger.info("Model loaded successfully.")

# Input schema
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

# Routes
@app.get("/")
def home():
    ##logger.info("Home endpoint hit.")
    return {"message": "Welcome to Churn Prediction API"}

@app.get("/health")
def health():
    #logger.info("Health check passed.")
    return {"status": "Healthy"}

@app.post("/predict")
def predict(data: CustomerData):
    #logger.info(f"Received data: {data}")
    
    features = np.array([[
        data.CreditScore,
        data.Gender,
        data.Age,
        data.Tenure,
        data.Balance,
        data.NumOfProducts,
        data.HasCrCard,
        data.IsActiveMember,
        data.EstimatedSalary,
        data.Geography_Germany,
        data.Geography_Spain
    ]])
    
    prediction = model.predict(features)[0]
    #logger.info(f"Prediction: {prediction}")
    return {"churn_prediction": int(prediction)}
