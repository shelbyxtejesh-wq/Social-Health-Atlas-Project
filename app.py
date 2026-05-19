import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(page_title="Epi-Predict | PHIDU", layout="wide")
st.title("🏥 Epi-Predict: Geospatial Forecasting Engine")
st.markdown("Predicting preventative hospitalization trajectories using Random Forest Regression.")
st.markdown("---")

# 2. Load Data and Train the Machine Learning Model
@st.cache_resource
def train_model():
    # Adjusted path for GitHub root directory
    df = pd.read_csv("longitudinal_health_data.csv")
    
    features = ['Year', 'Median_Age', 'SEIFA_Score', 'Clinics_per_10k', 'Diabetes_Pct']
    target = 'Preventable_Hosp_Rate'
    
    X = df[features]
    y = df[target]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return df, model

try:
    df, model = train_model()
except FileNotFoundError:
    st.error("Cannot find the data! Make sure longitudinal_health_data.csv is uploaded to your GitHub repository.")
    st.stop()

# 3. Build the Interactive Sidebar
st.sidebar.header("Policy Intervention Simulator")
st.sidebar.write("Adjust the parameters to see how health policies impact future hospitalizations.")

selected_lga = st.sidebar.selectbox("Select Target Region (LGA)", sorted(df['LGA_Name'].unique()))
funding_boost = st.sidebar.slider("Increase Regional Clinic Capacity (%)", min_value=0, max_value=100, value=0, step=5)

# 4. The Forecasting Engine
current_data = df[(df['LGA_Name'] == selected_lga) & (df['Year'] == 2024)].iloc[0]

future_years = [2025, 2026, 2027, 2028, 2029, 2030]
baseline_preds = []
intervention_preds = []

for year in future_years:
    sim_age = current_data['Median_Age'] + ((year - 2024) * 0.25)
    sim_seifa = current_data['SEIFA_Score']
    sim_diabetes = current_data['Diabetes_Pct'] + ((year - 2024) * 0.2)
    
    baseline_clinics = current_data['Clinics_per_10k']
    intervention_clinics = baseline_clinics * (1 + (funding_boost / 100.0))
    
    baseline_pred = model.predict([[year, sim_age, sim_seifa, baseline_clinics, sim_diabetes]])[0]
    intervention_pred = model.predict([[year, sim_age, sim_seifa, intervention_clinics, sim_diabetes]])[0]
    
    baseline_preds.append(baseline_pred)
    intervention_preds.append(intervention_pred)

# 5. Display the Results
col1, col2, col3 = st.columns(3)

col1.metric(label="Current Hospitalizations (2024)", value=int(current_data['Preventable_Hosp_Rate']))

base_diff = int(baseline_preds[-1] - current_data['Preventable_Hosp_Rate'])
col2.metric(label="Baseline Forecast (2030)", value=int(baseline_preds[-1]), delta=f"{base_diff} (If no action taken)", delta_color="inverse")

int_diff = int(intervention_preds[-1] - baseline_preds[-1])
col3.metric(label="Intervention Forecast (2030)", value=int(intervention_preds[-1]), delta=f"{int_diff} (Impact of Funding)", delta_color="inverse")

st.write("")
st.subheader(f"Hospitalization Trajectory: {selected_lga} (2025 - 2030)")

# Format data for the chart
chart_data = pd.DataFrame({
    'Year': future_years,
    'Baseline (Do Nothing)': baseline_preds,
    'Predicted with Intervention': intervention_preds
}).set_index('Year')

st.line_chart(chart_data, color=["#FF4B4B", "#00CC96"])
