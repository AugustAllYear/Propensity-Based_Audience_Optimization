"""Evaluate model and run six‑month simulation."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import roc_auc_score, classification_report
from src.data import generate_data, preprocess_data
from src.utils import load_config, setup_logging, ensure_dir, plot_roc_curve

logger = setup_logging()

def six_month_simulation(df, model, preprocessor, features, top_percent=0.3, random_months=3, total_months=6):
    """Simulate random vs model‑based targeting over months."""
    cumulative_opens = []
    strategy = ['random'] * random_months + ['model'] * (total_months - random_months)

    for month in range(total_months):
        month_data = df.sample(frac=1, replace=False).reset_index(drop=True)
        if strategy[month] == 'random':
            selected = month_data.sample(frac=top_percent, random_state=month)
        else:
            X_month = month_data[features]
            X_transformed = preprocessor.transform(X_month)
            probs = model.predict_proba(X_transformed)[:, 1]
            month_data = month_data.assign(pred_prob=probs)
            selected = month_data.sort_values('pred_prob', ascending=False).head(int(top_percent * len(month_data)))
        cumulative_opens.append(selected['opened'].sum())

    return cumulative_opens

def main():
    config = load_config()
    df = generate_data(
        n_customers=config['data']['n_customers'],
        random_state=config['data']['random_state'],
        sent_prob=config['data']['sent_prob']
    )

    # Load best model (Random Forest)
    model_path = os.path.join(config['paths']['models'], "random_forest.joblib")
    preprocessor_path = os.path.join(config['paths']['models'], "preprocessor.joblib")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Prepare test set
    train_df = df[df['sent'] == 1].copy()
    features = config['features']['numeric'] + config['features']['categorical']
    X_train, X_test, y_train, y_test, _ = preprocess_data(
        train_df,
        numeric_features=config['features']['numeric'],
        categorical_features=config['features']['categorical'],
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )

    # Evaluate
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"Model ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    # Plot ROC curve
    img_dir = config['paths']['images']
    ensure_dir(img_dir)
    plot_roc_curve(y_test, y_proba, save_path=os.path.join(img_dir, "roc_curve.png"))

    # Run simulation
    sim_opens = six_month_simulation(
        df, model, preprocessor, features,
        top_percent=config['targeting']['top_percent'],
        random_months=config['targeting']['random_initial_months'],
        total_months=config['targeting']['simulation_months']
    )
    improvement = (sum(sim_opens[3:]) / sum(sim_opens[:3]) - 1) * 100
    logger.info(f"Six‑month improvement: {improvement:.1f}%")

    # Plot simulation
    plt.figure(figsize=(8,5))
    plt.plot(range(1, config['targeting']['simulation_months']+1), sim_opens, marker='o')
    plt.axvline(x=config['targeting']['random_initial_months'] + 0.5, linestyle='--', color='gray')
    plt.xlabel('Month')
    plt.ylabel('Number of opens')
    plt.title('Cumulative Reach Over Six Months')
    plt.grid(True)
    plt.savefig(os.path.join(img_dir, "simulation.png"), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()