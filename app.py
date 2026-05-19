import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# 1. Page Configuration & Custom CSS
st.set_page_config(page_title="Epi-Predict | PHIDU", layout="wide", initial_sidebar_state="expanded")
st.title("🏥 Epi-Predict: Advanced Forecasting & ML Engine")
st.markdown("Predicting preventative hospitalization trajectories using ensemble machine learning.")
st.markdown("---")

# 2. Load Data and Train ML Model
@st.cache_resource
def train_model():
    df = pd.read_csv("longitudinal_health_data.csv")
    features = ['Year', 'Median_Age', 'SEIFA_Score', 'Clinics_per_10k', 'Diabetes_Pct']
    target = 'Preventable_Hosp_Rate'
    
    X = df[features]
    y = df[target]
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X, y)
    
    # Calculate Feature Importance for the UI
    importance = pd.DataFrame({
        'Feature': ['Time (Year)', 'Demographics (Age)', 'Socioeconomic (SEIFA)', 'Healthcare Access (Clinics)', 'Comorbidity (Diabetes)'],
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=True)
    
    return df, model, importance, features

try:
    df, model, importance, features = train_model()
except FileNotFoundError:
    st.error("Data missing. Ensure longitudinal_health_data.csv is in the repo.")
    st.stop()

# 3. Sidebar Controls
st.sidebar.header("⚙️ Policy Simulator")
selected_lga = st.sidebar.selectbox("Target Region (LGA)", sorted(df['LGA_Name'].unique()))
funding_boost = st.sidebar.slider("Increase Regional Clinic Capacity (%)", 0, 100, 0, 5)

st.sidebar.markdown("---")
st.sidebar.caption("Data Source: Synthetic PHIDU / ABS simulation.")
st.sidebar.caption("Algorithm: Random Forest Regressor")

# 4. Engine Logic
current_data = df[(df['LGA_Name'] == selected_lga) & (df['Year'] == 2024)].iloc[0]
future_years = [2025, 2026, 2027, 2028, 2029, 2030]
baseline_preds, intervention_preds = [], []

for year in future_years:
    sim_age = current_data['Median_Age'] + ((year - 2024) * 0.25)
    sim_seifa = current_data['SEIFA_Score']
    sim_diabetes = current_data['Diabetes_Pct'] + ((year - 2024) * 0.2)
    
    baseline_clinics = current_data['Clinics_per_10k']
    intervention_clinics = baseline_clinics * (1 + (funding_boost / 100.0))
    
    baseline_preds.append(model.predict([[year, sim_age, sim_seifa, baseline_clinics, sim_diabetes]])[0])
    intervention_preds.append(model.predict([[year, sim_age, sim_seifa, intervention_clinics, sim_diabetes]])[0])

# 5. UI Layout: Professional Tabs
tab1, tab2 = st.tabs(["📊 Forecasting Dashboard", "🧠 Under the Hood: ML Explainability"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Hospitalizations (2024)", int(current_data['Preventable_Hosp_Rate']))
    col2.metric("Baseline Forecast (2030)", int(baseline_preds[-1]), f"{int(baseline_preds[-1] - current_data['Preventable_Hosp_Rate'])} (No action)", delta_color="inverse")
    col3.metric("Intervention Forecast (2030)", int(intervention_preds[-1]), f"{int(intervention_preds[-1] - baseline_preds[-1])} (Impact)", delta_color="inverse")

    st.write("")
    
    # Advanced Plotly Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_years, y=baseline_preds, mode='lines+markers', name='Baseline Trajectory', line=dict(color='#FF4B4B', width=3, dash='dash')))
    fig.add_trace(go.Scatter(x=future_years, y=intervention_preds, mode='lines+markers', name='Intervention Trajectory', line=dict(color='#00CC96', width=4)))
    
    fig.update_layout(
        title=f"Predictive Trajectory: {selected_lga}",
        xaxis_title="Projection Year",
        yaxis_title="Preventable Hospitalizations (per 100k)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Algorithmic Decision Drivers")
    st.markdown("This chart illustrates which variables carry the most mathematical weight when the Random Forest algorithm predicts future hospitalizations. Understanding this allows policymakers to target the most impactful areas.")
    
    # Plotly Feature Importance Chart
    fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Blues')
    fig_imp.update_layout(xaxis_title="Relative Weight in ML Model", yaxis_title="", template="plotly_white")
    st.plotly_chart(fig_imp, use_container_width=True)
    
    with st.expander("View Raw Training Data"):
        st.dataframe(df.tail(50), use_container_width=True)
