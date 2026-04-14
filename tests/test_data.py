import pytest
from src.data import generate_data, preprocess_data
from src.utils import load_config

def test_generate_data():
    df = generate_data(n_customers=100)
    assert df.shape[0] == 100
    assert 'opened' in df.columns
    assert 'sent' in df.columns

def test_preprocess_data():
    config = load_config()
    df = generate_data(n_customers=500)
    train_df = df[df['sent'] == 1]
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        train_df,
        numeric_features=config['features']['numeric'],
        categorical_features=config['features']['categorical'],
        test_size=0.2
    )
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)