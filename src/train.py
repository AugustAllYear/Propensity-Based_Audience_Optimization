"""Train Random Forest and XGBoost models."""

import mlflow
import mlflow.sklearn
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from src.data import generate_data, preprocess_data
from src.utils import load_config, setup_logging, ensure_dir

logger = setup_logging()

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

    # Generate and preprocess data
    df = generate_data(
        n_customers=config['data']['n_customers'],
        random_state=config['data']['random_state'],
        sent_prob=config['data']['sent_prob']
    )
    # Use only customers who were sent the campaign
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
        y_proba = xgb_model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(xgb_model, "xgboost_model")
        logger.info(f"XGBoost ROC-AUC: {auc:.4f}")

    # Save models and preprocessor
    model_dir = config['paths']['models']
    ensure_dir(model_dir)
    joblib.dump(rf_model, os.path.join(model_dir, "random_forest.joblib"))
    joblib.dump(xgb_model, os.path.join(model_dir, "xgboost.joblib"))
    joblib.dump(preprocessor, os.path.join(model_dir, "preprocessor.joblib"))
    logger.info("Models and preprocessor saved.")

if __name__ == "__main__":
    main()
        