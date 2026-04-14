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
├── config/
│   └── config.yaml
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── notebooks/
│   └── propensity_educational.ipynb   (lightweight demo)
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── models/                             (saved models)
├── images/                             (plots)
├── requirements.txt
├── README.md
└── .gitignore
```

## Results
- The model identified recency (`last_purchase_days`) as the strongest predictor of opens.
- Targeting the top 30% of customers by predicted probability captures ~68% of all potential opens.
- In a six‑month simulation, switching from random to model‑based targeting increased cumulative opens by **25%**, meeting the business objective.
- Hyperparameter tuning improved ROC‑AUC from 0.781 to 0.794; XGBoost achieved 0.782.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook `propensity_educational.ipynb`.
4. (Optional) Replace the synthetic data with your own CSV file, ensuring column names and data types match.

## Continuation and Refinement Suggestions
- **A/B Test the Model**: Run a live experiment comparing the model’s top 30% against a random 30% control group to validate the lift.
- **Feature Engineering**: Incorporate additional features such as customer lifetime value, previous campaign engagement history (e.g., number of opens in last 3 months), time‑based features (day of week, season), and average response time.
- **Model Improvement**: Experiment with LightGBM, deeper hyperparameter tuning, or ensemble methods.
- **Automate Retraining**: Set up a scheduled pipeline (e.g., monthly) that ingests new campaign data, retrains the model, and updates the scoring system.
- **Deployment**: Package the model as a REST API using FastAPI, and integrate with marketing automation platforms (e.g., Salesforce Marketing Cloud, Braze).
- **Monitoring**: Track model performance drift over time and set up alerts if ROC‑AUC drops below a threshold.

**CI/CD**: Recommended platforms include GitHub Actions, GitLab CI, Jenkins, Azure DevOps, or CircleCI. For this project, GitHub Actions is used (free for public/private repos up to a limit). If data files exceed 14GB, consider upgrading to cloud storage (e.g., S3) and trigger jobs accordingly.

## License
MIT

## Contact
For questions, contact August Vollbrecht at augustvollbrecht@gmail.com