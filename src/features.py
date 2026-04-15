"""Feature engineering to improve performance, domain specificity, and prevent data leakage."""

import pandas as pd
import numpy as np

def add_velocity_features(df: pd.DataFrame, time_col='timestamp', id_col='user_id', window_hours=1):
    """
    Add rolling transaction count per user within the last `window_hours`.
    Requires datetime column.
    """
    df = df.sort_values([id_col, time_col]).copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df['tx_rolling'] = (
        df.groupby(id_col)[time_col]
        .transform(lambda x: x.rolling(window=f'{window_hours}h', closed='both').count())
    )
    return df

def add_amount_z_score(df: pd.DataFrame, amount_col='amount', id_col='user_id'):
    """Z-score of transaction amount per user."""
    df['amount_z_score'] = df.groupby(id_col)[amount_col].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    return df

def add_rolling_fraud_rate(df: pd.DataFrame, label_col='is_fraud', timestamp_col='timestamp', id_col='user_id', window=7):
    """
    Rolling fraud rate per user over a specific transaction window.
    """
    df = df.sort_values([id_col, timestamp_col]).copy()
    df['rolling_fraud_rate'] = (
        df.groupby(id_col)[label_col]
        .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
    )
    return df

def add_card_age(df: pd.DataFrame, card_issue_date_col='card_issue_date', ref_date=None):
    """Days since card was issued."""
    if ref_date is None:
        ref_date = pd.Timestamp.now()
    df['card_age_days'] = (ref_date - pd.to_datetime(df[card_issue_date_col])).dt.days
    return df

def engineer_features(df: pd.DataFrame, config: dict = None):
    """
    Apply feature engineering based on configuration.
    config keys: velocity (bool, window_hours), z_score (bool),
                 rolling_fraud (bool, window), card_age (bool)
    """
    if config is None:
        # Default: only z_score enabled (matches original model)
        config = {'velocity': False, 'z_score': True, 'rolling_fraud': False, 'card_age': False}

    if 'timestamp' in df.columns and config.get('velocity', False):
        window_hours = config.get('window_hours', 1)
        df = add_velocity_features(df, window_hours=window_hours)
    if config.get('z_score', False):
        df = add_amount_z_score(df)
    if config.get('rolling_fraud', False) and 'timestamp' in df.columns:
        window = config.get('window_transactions', 7)
        df = add_rolling_fraud_rate(df, window=window)
    if config.get('card_age', False) and 'card_issue_date' in df.columns:
        df = add_card_age(df)

    return df