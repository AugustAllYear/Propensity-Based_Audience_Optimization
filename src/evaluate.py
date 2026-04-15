"""Evaluate model and run six‑month simulation."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from src.data import generate_data, preprocess_data
from src.features import engineer_features
from src.utils import load_config, setup_logging, ensure_dir, plot_roc_curve

logger = setup_logging()

def six_month_simulation(df, model, preprocessor, features, top_percent=0.3, random_months=3, total_months=6, n_runs=10):
    """Simulate random vs model‑based targeting, averaged over multiple runs."""
    all_improvements = []
    last_sim_opens = None
    
    for run in range(n_runs):
        cumulative_opens = []
        strategy = ['random'] * random_months + ['model'] * (total_months - random_months)
        for month in range(total_months):
            month_data = df.sample(frac=1, replace=False).reset_index(drop=True)
            if strategy[month] == 'random':
                selected = month_data.sample(frac=top_percent, random_state=month + run * 100)
            else:
                X_month = month_data[features]
                X_transformed = preprocessor.transform(X_month)
                probs = model.predict_proba(X_transformed)[:, 1]
                month_data = month_data.assign(pred_prob=probs)
                selected = month_data.sort_values('pred_prob', ascending=False).head(int(top_percent * len(month_data)))
            cumulative_opens.append(selected['opened'].sum())
        
        random_total = sum(cumulative_opens[:random_months])
        model_total = sum(cumulative_opens[random_months:])
        if random_total > 0:
            improvement = (model_total / random_total - 1) * 100
            all_improvements.append(improvement)
        if run == n_runs - 1:
            last_sim_opens = cumulative_opens
    
    avg_improvement = np.mean(all_improvements) if all_improvements else 0.0
    return avg_improvement, last_sim_opens

def main():
    config = load_config()
    df = generate_data(
        n_customers=config['data']['n_customers'],
        random_state=config['data']['random_state'],
        sent_prob=config['data']['sent_prob']
    )

    # Apply feature engineering if enabled
    feat_config = config.get('features', {}).get('engineering', {})
    if feat_config.get('enabled', False):
        logger.info("Applying feature engineering...")
        df = engineer_features(df, config=feat_config)

    # Split data: training and simulation holdout (80/20)
    df_train, df_sim = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Training size: {len(df_train)}, Simulation size: {len(df_sim)}")

    # Load model and preprocessor (trained on full training set)
    model_path = os.path.join(config['paths']['models'], "random_forest.joblib")
    preprocessor_path = os.path.join(config['paths']['models'], "preprocessor.joblib")
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}. Run `python -m src.train` first.")
        return
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Evaluate on test set (from training split)
    train_df = df_train[df_train['sent'] == 1].copy()
    features = config['features']['numeric'] + config['features']['categorical']
    X_train, X_test, y_train, y_test, _ = preprocess_data(
        train_df,
        numeric_features=config['features']['numeric'],
        categorical_features=config['features']['categorical'],
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state']
    )
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"Model ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    # Plot ROC curve
    img_dir = config['paths']['images']
    ensure_dir(img_dir)
    plot_roc_curve(y_test, y_proba, save_path=os.path.join(img_dir, "roc_curve.png"))

    # Run simulation on holdout set (unseen data)
    improvement, sim_opens = six_month_simulation(
        df_sim, model, preprocessor, features,
        top_percent=config['targeting']['top_percent'],
        random_months=config['targeting']['random_initial_months'],
        total_months=config['targeting']['simulation_months'],
        n_runs=10
    )
    logger.info(f"Six‑month average improvement over 10 runs: {improvement:.1f}%")
    print(f"Average improvement: {improvement:.1f}%")
    print("Monthly opens (last run):", sim_opens)
    print("Random months total:", sum(sim_opens[:3]))
    print("Model months total:", sum(sim_opens[3:]))

    # Plot simulation
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(sim_opens)+1), sim_opens, marker='o')
    plt.axvline(x=config['targeting']['random_initial_months'] + 0.5, linestyle='--', color='gray', label='Switch to model')
    plt.xlabel('Month')
    plt.ylabel('Number of opens')
    plt.title('Cumulative Reach Over Six Months (Holdout Data)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(img_dir, "simulation.png"), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()