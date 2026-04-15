from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from src.predict import load_model, predict
from src.utils import load_config

app = FastAPI(title="Propensity Prediction API")
config = load_config()
model, preprocessor = load_model()

class  CustomerData(BaseModel):
    age: int
    income: int
    tenure: int
    last_purchase_days: int
    avg_order_value: int
    campaign_channel: str
    campaign_type: str

class PredictionResponse(BaseModel):
    open_probability: float

@app.post("/predict", response_model=PredictionResponse)
def predict_single(customer: CustomerData):
    df = pd.DataFrame([customer.dict()])
    prob = predict(df, model, preprocessor, config)[0]
    return {"open_probability": float(prob)}

@app.post("/predict_batch")
def predict_batch(customers: list[CustomerData]):
    df = pd.DataFrame([c.dict() for c in customers])
    probs = predict(df, model, preprocessor, config)
    return {"probabilities": probs.tolist()}

@app.get("/health")
def health():
    return {"status": "ok"}