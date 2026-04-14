"""Data generation and preprocessing."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def generate_data(n_customers=5000, random_state=42, sent_prob=0.8):
    """Generate synthetic customer campaign data."""
    np.random.seed(random_state)
    data = {
        'customer_id': range(1, n_customers+1),
        'age': np.random.randint(18, 70, n_customers),
        'income': np.random.normal(50000, 15000, n_customers).astype(int),
        'tenure': np.random.randint(1, 120, n_customers),
        'last_purchase_days': np.random.randint(1, 365, n_customers),
        'avg_order_value': np.random.normal(100, 30, n_customers).clip(20, 300).astype(int),
        'campaign_channel': np.random.choice(['email', 'social', 'push'], n_customers),
        'campaign_type': np.random.choice(['promotional', 'informational', 'loyalty'], n_customers),
        'sent': np.random.choice([0,1], n_customers, p=[1-sent_prob, sent_prob]),
    }
    df = pd.DataFrame(data)

    def generate_opened(row):
        prob = 0.1
        if row['last_purchase_days'] < 60:
            prob += 0.2
        if row['income'] > 60000:
            prob += 0.1
        if row['campaign_channel'] == 'email':
            prob += 0.15
        if row['campaign_type'] == 'promotional':
            prob += 0.1
        prob = min(prob, 0.9)
        return np.random.binomial(1, prob)

    df['opened'] = df.apply(generate_opened, axis=1)
    return df

def preprocess_data(df, numeric_features, categorical_features, test_size=0.2, random_state=42, fit_preprocessor=True):
    """Split, scale, encode."""
    X = df[numeric_features + categorical_features]
    y = df['opened']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    if fit_preprocessor:
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
    else:
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)

    return X_train, X_test, y_train, y_test, preprocessor