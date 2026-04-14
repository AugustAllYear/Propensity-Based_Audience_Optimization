"""Predict open probability for new customers."""

import pandas as pd
import joblib
import os
from src.utils import load_config

def load_model(model_path=None, preprocessor_path=None):
    config = load_config()
    if model_path is None:
        model_path = os.path.join(config['paths']['models'], "random_forest.joblib")
    if preprocessor_path is None:
        preprocessor_path = os.path.join(config['paths']['models'], "preprocessor.joblib")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def predict(new_data: pd.DataFrame, model, preprocessor, config):
    features = config['features']['numeric'] + config['features']['categorical']
    X = new_data[features]
    X_transformed = preprocessor.transform(X)
    probs = model.predict_proba(X_transformed)[:, 1]
    return probs

if __name__ == "__main__":
    from src.data import generate_data
    config = load_config()
    df = generate_data(n_customers=100)
    model, preprocessor = load_model()
    probs = predict(df, model, preprocessor, config)
    print("Sample probabilities:", probs[:5])