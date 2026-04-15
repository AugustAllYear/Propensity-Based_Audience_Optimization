# Propensity‑Based Audience Optimization

## Project Overview
This project develops a machine learning model to optimize email marketing campaigns. By predicting which customers are most likely to engage (open an email), the marketing team can target a smaller, higher‑potential audience, increasing overall reach while keeping send volume constant. The solution is replicable and can be integrated into a monthly campaign workflow.

## Business Problem
The company historically sent campaigns to its entire customer database, resulting in low open rates and wasted marketing spend. The goal was to increase the number of opens by **25% within six months** without increasing send volume.

## Data
Historical campaign data includes:
- **Customer demographics**: age, income, tenure (months), days since last purchase, average order value
- **Campaign attributes**: channel (email, social, push), type (promotional, informational, loyalty)
- **Engagement flag**: whether the customer opened the email (target variable)

The dataset is synthetically generated for demonstration; the methodology was applied to real customer data.

## Methodology
1. **Exploratory Data Analysis** – Visualised feature distributions and relationships with the target.
2. **Preprocessing** – Scaled numerical features and one‑hot encoded categorical variables.
3. **Modeling** – Trained a Random Forest classifier (baseline), tuned hyperparameters with GridSearchCV, and compared with XGBoost.
4. **Evaluation** – Used ROC‑AUC, precision, recall, and business‑oriented simulations.
5. **Simulation** – Compared random targeting with model‑based targeting over six months to quantify business impact.

**MLflow Tracking**: All experiments are automatically logged to the local `mlruns/` directory. Run `mlflow ui` to explore.

## Project Structure

```
propensity_optimization/
├── .github/workflows/
│ └── retrain.yml # Weekly model retraining (GitHub Actions)
├── config/
│ └── config.yaml # All configuration parameters
├── src/
│ ├── init.py
│ ├── data.py # Data generation & preprocessing
│ ├── features.py 
│ ├── train.py # Model training (RF, XGBoost)
│ ├── evaluate.py # Evaluation & simulation
│ ├── predict.py # Prediction functions
│ └── utils.py # Logging, plotting, config loading
├── tests/
│ ├── test_data.py
│ └── test_model.py
├── notebooks/
│ └── propensity_educational.ipynb
├── app.py # Streamlit dashboard
├── api.py # FastAPI batch prediction service
├── requirements.txt
├── .gitignore
└── README.md
```

## Results
- **Most important feature**: `last_purchase_days` (recency)
- **Targeting top 30%** captures ~68% of all potential opens.
- **Six‑month simulation** shows a **25% increase** in cumulative opens after switching from random to model‑based targeting.
- **ROC‑AUC improved** from 0.781 (baseline) to 0.794 (tuned Random Forest); XGBoost achieved 0.782.

## Setup

### Prerequisites
- Python 3.11
- Git

### Installation
```bash
    git clone https://github.com/yourusername/propensity_optimization.git
    cd propensity_optimization
    python -m venv venv
    source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install -r requirements.txt
```

### Configuration

All parameters are centralised in `config/config.yaml`. You can adjust data size, model hyperparameters, targeting thresholds, cost‑benefit defaults, and simulation settings without touching the code.

### Optional Feature Engineering

The codebase includes advanced feature engineering functions (`src/features.py`) that can generate additional predictors such as:

- **Transaction velocity** (rolling count per user over time)
- **Amount z‑score** (per‑user standardised transaction amount)
- **Rolling fraud rate** (if historical fraud labels exist)
- **Card age** (days since card issuance)

These features are **disabled by default** (`enabled: false` in `config/config.yaml` under `features.engineering`) because the synthetic data lacks the necessary columns (`timestamp`, `user_id`, `card_issue_date`, `is_fraud`). To enable them for your own dataset, set `enabled: true` and ensure your data includes the required columns. See `src/features.py` for implementation details.


### Usage

#### Evaluation, Train and run simulation
```
    from src.predict import load_model, predict
    import pandas as pd
    model, preprocessor = load_model()
    new_data = pd.read_csv("new_customers.csv")
    probs = predict(new_data, model, preprocessor, config)
```

**Train**

```bash
    python -m src.train
```

**Evaluate**

```bash
    python -m src.evaluate
```

**Predict**

```bash
    src.predict.predict()
```

*New data*
```python
    from src.predict import load_model, predict
    import pandas as pd
    model, preprocessor = load_model()
    new_data = pd.read_csv("new_customers.csv")
    probs = predict(new_data, model, preprocessor, config)
```

#### Run tests
```bash
    pytest tests/
```

#### Launch Dashboard

After training the models (`python -m src.train`), launch the interactive dashboard:

```bash
streamlit run dashboard.py
```

#### Start Prediction API
```bash
    uvicorn api:app --reload --port 8000
```

#### MLflow UI
```bash
    mlflow ui
```

### Dashboard Features

The dashboard allows you to:
- Upload customer CSV or use synthetic data.
- Score customers and see top X% most likely to open.
- View six‑month simulation chart (if ground truth exists).
- SHAP explanations for individual predictions.
- Feature importance bar chart.
- Cost‑benefit analysis with adjustable inputs.
- A/B test simulation with statistical significance.
- Model performance over time (from MLflow history).

### Access the services
Streamlit dashboard – Open your browser to http://localhost:8501 (it will open automatically if Streamlit is configured, but you can manually go there).

MLflow UI – Open http://localhost:5000

FastAPI – Open http://localhost:8000/docs for interactive API documentation.

### Integration Notes
- SHAP works with tree‑based models (Random Forest, XGBoost). For large datasets, SHAP is limited to the first 5 customers to maintain performance.
- FastAPI is free and open‑source. Run it alongside the dashboard for batch predictions.
- GitHub Actions (.github/workflows/retrain.yml) retrains the model every Sunday. For real data, adapt src/train.py to load from a fixed location (e.g., S3, database) and use secrets for credentials.
- MLflow performance chart requires logged runs with experiment name Propensity_Optimization. It will be empty until you train at least once.
- Cost‑benefit defaults are stored in config/config.yaml under the cost_benefit section.

### Production Considerations
1. Model Registry – Use MLflow Model Registry to promote models from staging to production.

- **Model Serialization**: For enhanced security, replace `mlflow.sklearn.log_model()` with `mlflow.skops.log_model()` (requires `skops`). This avoids pickle‑based vulnerabilities.

2. Automated Retraining – Schedule src/train.py weekly (already set up via GitHub Actions).

3. Data Validation – Implement schema checks (column names, types) before training/prediction.

4. Secrets Management – Never hardcode API keys; use environment variables or a secrets manager.

5. Monitoring – Track prediction drift (PSI) and model performance (ROC‑AUC) over time; set alerts for degradation.

6. Deployment – Package the model as a REST API (FastAPI) for real‑time scoring; the dashboard is for batch exploration only.

### FastAPI Prediction Service
The API (`api.py`) runs independently of the dashboard. To use it:
1. Train the model first: `python -m src.train`
2. Start the server: `uvicorn api:app --reload --port 8000`
3. Send POST requests to `/predict` (single) or `/predict_batch` (multiple).

The API is free and open‑source. You can deploy it on any cloud platform (e.g., Heroku, AWS, GCP) or run locally.

### Cost‑Benefit Defaults
Default values for cost per email, conversion rate, and average order value are stored in `config/config.yaml` under the `cost_benefit` section. Users can override them in the dashboard sidebar.

## Production Considerations

1. **Model Registry**: Use MLflow Model Registry to manage model versions (stage "Staging" → "Production").
2. **Automated Retraining**: Schedule `src/train.py` weekly via cron or GitHub Actions.
3. **Data Validation**: Implement schema checks (column names, types) before training/prediction – see `src/data.py`.
4. **Secrets Management**: Never hardcode API keys or database URIs; use environment variables or a secrets manager.
5. **Monitoring**: Track prediction drift (PSI) and model performance (ROC‑AUC) over time. Set alerts for degradation.
6. **Deployment**: Package the model as a REST API (FastAPI) for real‑time scoring. The dashboard is for batch exploration only.
    
## Continuation and Refinement Suggestions
- **A/B Test the Model**: Run a live experiment comparing the model’s top 30% against a random 30% control group to validate the lift.
- **Feature Engineering**: Incorporate additional features such as customer lifetime value, previous campaign engagement history (e.g., number of opens in last 3 months), time‑based features (day of week, season), and average response time.
- **Model Improvement**: Experiment with LightGBM, deeper hyperparameter tuning, or ensemble methods.
- **Automate Retraining**: Set up a scheduled pipeline (e.g., monthly) that ingests new campaign data, retrains the model, and updates the scoring system.
- **Deployment**: Package the model as a REST API using FastAPI, and integrate with marketing automation platforms (e.g., Salesforce Marketing Cloud, Braze).
- **Monitoring**: Track model performance drift over time and set up alerts if ROC‑AUC drops below a threshold.

**CI/CD**: Recommended platforms include GitHub Actions, GitLab GitHub Actions is used for CI/CD (free for public/private repos up to limits). If data files exceed 14GB, consider upgrading to cloud storage (e.g., S3) and triggering jobs externally.

## Related Project: Sentinel_AI Fraud Detection

This project inspired and was adapted to Sentinel_AI, an end‑to‑end fraud detection system. Both share the same production‑ready architecture: configuration‑driven scripts, MLflow tracking, GitHub Actions CI/CD, and a Streamlit dashboard.

## License
MIT

## Contact
For questions, contact August Vollbrecht at augustvollbrecht@gmail.com