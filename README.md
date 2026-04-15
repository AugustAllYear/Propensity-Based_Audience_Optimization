# Propensity‑Based Audience Optimization

## Project Overview
This project develops a machine learning model to optimize email marketing campaigns. By predicting which customers are most likely to engage (open an email), the marketing team can target a smaller, higher‑potential audience, increasing overall reach while keeping send volume constant. The solution is designed to be replicable and can be integrated into a monthly campaign workflow.

## Business Problem
The company historically sent campaigns to its entire customer database, resulting in low open rates and wasted marketing spend. The goal was to use data‑driven targeting to increase the number of opens by 25% within six months, without increasing the number of emails sent.

## Data
We used historical campaign data containing:
- Customer demographics: age, income, tenure (months), days since last purchase, average order value
- Campaign attributes: channel (email, social, push), type (promotional, informational, loyalty)
- Engagement flag: whether the customer opened the email (target variable)

The dataset was synthetically generated for demonstration; the methodology was applied to real customer data.

## Methodology
1. **Exploratory Data Analysis**: Visualized feature distributions and relationships with the target. 
2. **Preprocessing**: Scaled numerical features and one‑hot encoded categorical variables.
3. **Modeling**: Trained a Random Forest classifier (baseline), tuned hyperparameters with GridSearchCV, and compared with XGBoost.
4. **Evaluation**: Used ROC‑AUC, precision, recall, and business‑oriented simulations.
5. **Simulation**: Compared random targeting with model‑based targeting over six months to quantify business impact.

**MLflow Tracking**: All experiments (baseline, tuned Random Forest, XGBoost) are automatically logged to the local `mlruns/` directory. To view the UI, run `mlflow ui`.

## Project Structure

```
propensity_optimization/
├── .github/
│ └── workflows/
│ └── retrain.yml # Weekly model retraining (GitHub Actions)
├── config/
│ └── config.yaml # All configuration parameters
├── src/
│ ├── init.py
│ ├── data.py # Data generation & preprocessing
│ ├── features.py # Feature engineering (if any)
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
- The model identified recency (`last_purchase_days`) as the strongest predictor of opens.
- Targeting the top 30% of customers by predicted probability captures ~68% of all potential opens.
- In a six‑month simulation, switching from random to model‑based targeting increased cumulative opens by **25%**, meeting the business objective.
- Hyperparameter tuning improved ROC‑AUC from 0.781 to 0.794; XGBoost achieved 0.782.


## Setup
```bash
    git clone ...
    cd propensity_optimization
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
```
### Configuration

All parameters are centralized in `config/config.yaml`. You can adjust data size, model hyperparameters, targeting thresholds, and simulation settings without touching the code.

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
    python -m src.trian
```

**Evaluate**

```bash
    python -m
```

**Predict**

```bash
    src.predict.predict()
```

#### MLflow Tracking
```bash
    mlflow ui
```

#### Run tests
```bash
    pytest tests/
```


## Running the Dashboard

After training the models (`python -m src.train`), launch the interactive dashboard:

```bash
streamlit run dashboard.py
```

The dashboard allows you to:
- Upload your own customer CSV or use synthetic data.
- Score customers and see the top X% most likely to open.
- View the six‑month simulation chart (if ground truth labels exist).
- Explore feature importances (for Random Forest).

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

**CI/CD**: Recommended platforms include GitHub Actions, GitLab CI, Jenkins, Azure DevOps, or CircleCI. For this project, GitHub Actions is used (free for public/private repos up to a limit). If data files exceed 14GB, consider upgrading to cloud storage (e.g., S3) and trigger jobs accordingly.

## Related Project: Sentinel_AI Fraud Detection

This propensity optimization pipeline inspired and adapted to **[Sentinel_AI](https://github.com/AugustAllYear/Sentinel_AI)**, an end‑to‑end fraud detection system. Sentinel_AI uses similar architectural patterns: configuration‑driven scripts, MLflow tracking, CI/CD with GitHub Actions, and a Streamlit dashboard. The main differences are the domain (marketing vs. fraud) and the evaluation focus (lift in opens vs. fraud capture rate). Both projects share the same production‑ready structure.

## License
MIT

## Contact
For questions, contact August Vollbrecht at augustvollbrecht@gmail.com