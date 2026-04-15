# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import scipy.sparse as sp
from scipy.stats import chi2_contingency
from mlflow.tracking import MlflowClient
from src.data import generate_data
from src.utils import load_config
from src.predict import load_model, predict
from src.evaluate import six_month_simulation

# Load config
config = load_config()

# Dashboard config
dashboard_cfg = config.get('dashboard', {})

st.set_page_config(
    page_title=dashboard_cfg.get('title', "Propensity Dashboard"),
    layout=dashboard_cfg.get('page_layout', 'wide')
)

st.title("Propensity‑Based Audience Optimization Dashboard")

# Load model and preprocessor
model, preprocessor = load_model()

# Sidebar
st.sidebar.header("Options")
data_source = st.sidebar.selectbox("Data Source", ["Synthetic Demo", "Upload CSV"])
default_top = dashboard_cfg.get('default_top_percent', config['targeting']['top_percent'])
top_percent = st.sidebar.slider("Top % to target", 0.1, 0.5, default_top, 0.05)

# Cost-benefit inputs (from config - user update in config)
cost_cfg = config.get('cost_benefit', {})
default_cost = cost_cfg.get('default_cost_per_email', 0.05)
default_conversion = cost_cfg.get('default_conversion_rate', 0.1)
default_avg_order = cost_cfg.get('default_avg_order_value', 100)

st.sidebar.subheader("Cost‑Benefit Analysis")
cost_per_email = st.sidebar.number_input("Cost per email sent ($)", 0.01, 1.0, default_cost, step=0.01)
conversion_rate = st.sidebar.slider("Conversion rate (given open)", 0.0, 1.0, default_conversion, 0.01)
avg_order_value = st.sidebar.number_input("Average order value ($)", 10, 1000, default_avg_order, step=10)

# Load data
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload customer data (CSV)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.warning("Please upload a file")
        st.stop()
else:
    df = generate_data(
        n_customers=config['data']['n_customers'],
        random_state=config['data']['random_state'],
        sent_prob=config['data']['sent_prob']
    )

# Predict
X = df[config['features']['numeric'] + config['features']['categorical']]
probs = predict(df, model, preprocessor, config)
df['pred_open_prob'] = probs

# Sort and get top customers
df_sorted = df.sort_values('pred_open_prob', ascending=False)
top_n = int(top_percent * len(df))
top_customers = df_sorted.head(top_n)

st.subheader(f"Top {top_percent*100:.0f}% Customers Most Likely to Open")
st.dataframe(top_customers[['customer_id', 'pred_open_prob'] + config['features']['numeric'] + config['features']['categorical']].head(20))

# If ground truth exists
if 'opened' in df.columns:
    actual_opens_full = df['opened'].sum()
    actual_opens_top = top_customers['opened'].sum()
    st.metric("Opens captured in top %", f"{actual_opens_top}/{actual_opens_full} ({actual_opens_top/actual_opens_full:.1%})")

    # Six‑month simulation
    improvement, sim_opens = six_month_simulation(
    df, model, preprocessor,
    config['features']['numeric'] + config['features']['categorical'],
    top_percent=config['targeting']['top_percent'],
    random_months=config['targeting']['random_initial_months'],
    total_months=config['targeting']['simulation_months']
)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(sim_opens)+1), sim_opens, marker='o')
    ax.axvline(x=config['targeting']['random_initial_months']+0.5, linestyle='--', color='gray')
    ax.set_xlabel("Month")
    ax.set_ylabel("Opens")
    ax.set_title("Six‑Month Simulation")
    ax.grid(True)
    st.pyplot(fig)

# Cost-benefit analysis (if ground truth or predicted)
if st.checkbox("Show cost‑benefit analysis"):
    sends = top_n
    expected_opens = top_customers['opened'].sum() if 'opened' in df else top_customers['pred_open_prob'].sum()
    expected_conversions = expected_opens * conversion_rate
    revenue = expected_conversions * avg_order_value
    cost = sends * cost_per_email
    profit = revenue - cost
    st.metric("Expected profit", f"${profit:,.2f}")
    st.caption(f"Based on {sends} sends, {expected_opens:.0f} expected opens, {expected_conversions:.0f} conversions.")

# SHAP explanations
st.subheader("Model Explanations (SHAP)")
if st.checkbox("Show SHAP explanations for first 5 customers"):
    sample = df[config['features']['numeric'] + config['features']['categorical']].head(5)
    X_sample = preprocessor.transform(sample)
    if sp.issparse(X_sample):
        X_sample = X_sample.toarray()
    feature_names = preprocessor.get_feature_names_out()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)[1]  # class 1 (open)
    shap.initjs()
    st.write("#### Force plot for first customer")
    shap.force_plot(explainer.expected_value[1], shap_values[0], X_sample[0],
                    feature_names=feature_names, matplotlib=True, show=False)
    plt.savefig("shap_force.png", bbox_inches='tight')
    st.image("shap_force.png")

# Feature importance
if hasattr(model, 'feature_importances_'):
    st.subheader("Feature Importances")
    feature_names = config['features']['numeric'] + config['features']['categorical']
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    st.bar_chart(fi_df.set_index('feature'))

# Model performance over time
def get_performance_history(experiment_name="Propensity_Optimization"):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return pd.DataFrame()
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time ASC"])
    data = []
    for run in runs:
        data.append({
            "date": run.info.start_time,
            "run_name": run.data.tags.get("mlflow.runName", "unknown"),
            "roc_auc": run.data.metrics.get("roc_auc", None)
        })
    return pd.DataFrame(data)

if st.checkbox("Show model performance over time"):
    perf_df = get_performance_history()
    if not perf_df.empty:
        st.line_chart(perf_df.set_index("date")["roc_auc"])
    else:
        st.info("No MLflow runs found. Train a model first.")

# A/B test simulation
if st.button("Run A/B Test Simulation"):
    random_sample = df.sample(frac=top_percent, random_state=42)
    model_sample = df_sorted.head(top_n)
    if 'opened' in df.columns:
        control_opens = random_sample['opened'].sum()
        model_opens = model_sample['opened'].sum()
        control_size = len(random_sample)
        model_size = len(model_sample)
        table = [[control_opens, control_size - control_opens],
                 [model_opens, model_size - model_opens]]
        chi2, p, dof, expected = chi2_contingency(table)
        st.metric("Model lift", f"{(model_opens/control_opens - 1):.1%}")
        st.write(f"Chi‑square p‑value: {p:.4f}")
        if p < 0.05:
            st.success("Statistically significant improvement (p < 0.05)")
        else:
            st.warning("Not statistically significant")
    else:
        st.info("Ground truth labels not available. Train model with historical data to run A/B test.")