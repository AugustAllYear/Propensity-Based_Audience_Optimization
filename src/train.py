"""Train Random Forest and XGBoost models."""

import mlflow
import mlflow.sklearn
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.data import generate_data, preprocess_data
from src.features import engineer_features
from src.utils import setup_logging, load_config
from xgboost import XGBClassifier

logger = setup_logging()

def load_real_data(file_path: str):
    df = pd.read_csv(file_path)
    required_cols = ['age', 'income', 'tenure', 'last_purchase_days', 'avg_order_value',
                     'campaign_channel', 'campaign_type', 'opened', 'sent']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df

# --- Exported training functions ---
def train_random_forest(X_train, y_train, config):
    rf_cfg = config['model']['random_forest']
    model = RandomForestClassifier(
        n_estimators=rf_cfg['n_estimators'],
        random_state=rf_cfg['random_state'],
        class_weight=rf_cfg['class_weight']
    )
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, config):
    xgb_cfg = config['model']['xgboost']
    model = XGBClassifier(
        n_estimators=xgb_cfg['n_estimators'],
        random_state=xgb_cfg['random_state'],
        eval_metric=xgb_cfg['eval_metric']
    )
    model.fit(X_train, y_train)
    return model

def main():
    config = load_config()
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    # Load or generate data
    data_path = None  # Modify to accept argument
    if data_path is not None:
        logger.info(f"Loading real data from {data_path}")
        df = load_real_data(data_path)
    else:
        logger.info("Generating synthetic data.")
        df = generate_data(
            n_customers=config['data']['n_customers'],
            random_state=config['data']['random_state'],
            sent_prob=config['data']['sent_prob']
        )

    # Optional feature engineering
    feat_config = config.get('features', {}).get('engineering', {})
    if feat_config.get('enabled', False):
        logger.info("Applying feature engineering...")
        df = engineer_features(df, config=feat_config)

    # Preprocess
    train_df = df[df['sent'] == 1].copy()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
        train_df,
        numeric_features=config['features']['numeric'],
        categorical_features=config['features']['categorical'],
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )

    # Train Random Forest
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        rf_model = train_random_forest(X_train, y_train, config)
        y_proba = rf_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(rf_model, "random_forest_model")
        logger.info(f"Random Forest ROC-AUC: {auc:.4f}")

    # Train XGBoost
    with mlflow.start_run(run_name="XGBoost_Baseline"):
        xgb_model = train_xgboost(X_train, y_train, config)
        y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
        auc_xgb = roc_auc_score(y_test, y_proba_xgb)
        mlflow.log_metric("roc_auc", auc_xgb)
        mlflow.sklearn.log_model(xgb_model, "xgboost_model")
        logger.info(f"XGBoost ROC-AUC: {auc_xgb:.4f}")

    # Save models and preprocessor
    model_dir = config['paths']['models']
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(rf_model, os.path.join(model_dir, "random_forest.joblib"))
    joblib.dump(xgb_model, os.path.join(model_dir, "xgboost.joblib"))
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))
    logger.info("Models and preprocessor saved.")

if __name__ == "__main__":
    main()

