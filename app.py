# dashboard.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from src.data import generate_data
from src.utils import load_config
from src.predict import load_model, predict

st.set_page_config(page_title="Propensity Dashboard", layout="wide")
st.title("📧 Propensity‑Based Audience Optimization Dashboard")

config = load_config()
model, preprocessor = load_model()

# Sidebar
st.sidebar.header("Options")
data_source = st.sidebar.selectbox("Data Source", ["Synthetic Demo", "Upload CSV"])
top_percent = st.sidebar.slider("Top % to target", 0.1, 0.5, config['targeting']['top_percent'], 0.05)

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

# Show top % recommendations
df_sorted = df.sort_values('pred_open_prob', ascending=False)
top_n = int(top_percent * len(df))
top_customers = df_sorted.head(top_n)

st.subheader(f"🎯 Top {top_percent*100:.0f}% Customers Most Likely to Open")
st.dataframe(top_customers[['customer_id', 'pred_open_prob'] + config['features']['numeric'] + config['features']['categorical']].head(20))

# Actual opens captured (if ground truth exists)
if 'opened' in df.columns:
    actual_opens_full = df['opened'].sum()
    actual_opens_top = top_customers['opened'].sum()
    st.metric("Opens captured in top %", f"{actual_opens_top}/{actual_opens_full} ({actual_opens_top/actual_opens_full:.1%})")

# Simulation (if ground truth exists)
if 'opened' in df.columns:
    from src.evaluate import six_month_simulation
    sim_opens = six_month_simulation(
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

# Feature importance (if model is RandomForest)
if hasattr(model, 'feature_importances_'):
    st.subheader("Feature Importances")
    feature_names = config['features']['numeric'] + config['features']['categorical']
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values('importance', ascending=False)
    st.bar_chart(fi_df.set_index('feature'))