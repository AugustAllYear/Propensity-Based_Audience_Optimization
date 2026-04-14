import pytest
from src.train import train_random_forest, train_xgboost
from src.data import generate_data, preprocess_data
from src.utils import load_config

def test_random_forest_training():
    config = load_config()
    df = generate_data(n_customers=500)
    train_df = df[df['sent'] == 1]
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        train_df,
        numeric_features=config['features']['numeric'],
        categorical_features=config['features']['categorical']
    )
    model = train_random_forest(X_train, y_train, config)
    assert hasattr(model, 'predict_proba')

def test_xgboost_training():
    config = load_config()
    df = generate_data(n_customers=500)
    train_df = df[df['sent'] == 1]
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        train_df,
        numeric_features=config['features']['numeric'],
        categorical_features=config['features']['categorical']
    )
    model = train_xgboost(X_train, y_train, config)
    assert hasattr(model, 'predict_proba')