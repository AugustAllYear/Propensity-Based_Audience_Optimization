"""Predict open probability for new customers."""

import pandas as pd
import joblib
import os
from src.utils import load_config

def load_model(model_path=None, preprocessor_path=None):
    """Load trained model and preprocessor from paths (or defaults from config)."""
    config = load_config()
    if model_path is None:
        model_path = os.path.join(config['paths']['models'], "random_forest.joblib")
    if preprocessor_path is None:
        preprocessor_path = os.path.join(config['paths']['models'], "preprocessor.joblib")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def predict(new_data: pd.DataFrame, model, preprocessor, config):
    """
    Predict open probabilities for new customers.
    
    Args:
        new_data: DataFrame with same columns as training data.
        model: Trained classifier (Random Forest or XGBoost).
        preprocessor: Fitted ColumnTransformer.
        config: Loaded configuration dictionary.
    
    Returns:
        Array of predicted probabilities (0 to 1).
    """
    features = config['features']['numeric'] + config['features']['categorical']
    # Ensure required columns exist
    missing = set(features) - set(new_data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    X = new_data[features]
    X_transformed = preprocessor.transform(X)
    probs = model.predict_proba(X_transformed)[:, 1]
    return probs

if __name__ == "__main__":
    # Quick test: generate synthetic data and predict
    from src.data import generate_data
    config = load_config()
    df = generate_data(n_customers=100)
    model, preprocessor = load_model()
    probs = predict(df, model, preprocessor, config)
    print("Sample probabilities:", probs[:5])