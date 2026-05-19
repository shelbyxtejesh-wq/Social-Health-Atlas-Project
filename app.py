import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import plotly.express as px
import warnings
import time
warnings.filterwarnings('ignore')

# 1. Premium Executive UI CSS (Matte Black, Metallic Gold, Deep Red)
st.set_page_config(page_title="Epi-Predict | AI Copilot", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #E0E0E0; }
    h1, h2, h3, h4 { color: #D4AF37 !important; font-family: 'Helvetica Neue', sans-serif; }
    .stMetricValue { color: #FFFFFF !important; }
    div[data-testid="stSidebar"] { background-color: #1A1A1A; border-right: 1px solid #D4AF37; }
    .stChatFloatingInputContainer { background-color: #1A1A1A !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Epi-Predict: AI Copilot & Digital Twin")
st.markdown("Advanced ensemble forecasting with embedded Generative AI policy analysis.")
st.markdown("---")

# 2. Data Engineering & ML Training
@st.cache_resource
def load_and_train():
    df = pd.read_csv("longitudinal_health_data.csv")
    features = ['Year', 'Median_Age', 'SEIFA_Score', 'Clinics_per_10k', 'Diabetes_Pct']
    target = 'Preventable_Hosp_Rate'
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(df[features], df[target])
    
    return df, model, features

try:
    df, model, features = load_and_train()
except Exception:
    st.error("Error loading data. Check repository.")
    st.stop()

# 3. Sidebar UI
st.sidebar.header("⚙️ Policy Constraints")
selected_lga = st.sidebar.selectbox("Target Region (LGA)", sorted(df['LGA_Name'].unique()))
funding_boost = st.sidebar.slider("Increase Regional Clinic Capacity (%)", 0, 100, 0, 5)

# 4. Engine Logic
current_data = df[(df['LGA_Name'] == selected_lga) & (df['Year'] == 2024)].iloc[0]
future_years = [2025, 2026, 2027, 2028, 2029, 2030]
baseline_preds, intervention_preds = [], []

for year in future_years:
    sim_age = current_data['Median_Age'] + ((year - 2024) * 0.25)
    sim_seifa, sim_diabetes = current_data['SEIFA_Score'], current_data['Diabetes_Pct'] + ((year - 2024) * 0.2)
    base_clinics = current_data['Clinics_per_10k']
    int_clinics = base_clinics * (1 + (funding_boost / 100.0))
    
    baseline_preds.append(model.predict([[year, sim_age, sim_seifa, base_clinics, sim_diabetes]])[0])
    intervention_preds.append(model.predict([[year, sim_age, sim_seifa, int_clinics, sim_diabetes]])[0])

# 5. UI Architecture
tab1, tab2 = st.tabs(["📊 Digital Twin Forecast", "💬 AI Policy Copilot"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Hospitalizations (2024)", int(current_data['Preventable_Hosp_Rate']))
    col2.metric("Baseline Forecast (2030)", int(baseline_preds[-1]), f"{int(baseline_preds[-1] - current_data['Preventable_Hosp_Rate'])} (No action)", delta_color="inverse")
    col3.metric("Intervention Forecast (2030)", int(intervention_preds[-1]), f"{int(intervention_preds[-1] - baseline_preds[-1])} (Impact)", delta_color="inverse")
    
    # Premium Styled Chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_years, y=baseline_preds, mode='lines+markers', name='Baseline Trajectory', line=dict(color='#8B0000', dash='dash', width=3)))
    fig.add_trace(go.Scatter(x=future_years, y=intervention_preds, mode='lines+markers', name='Intervention Trajectory', line=dict(color='#D4AF37', width=4)))
    fig.update_layout(title=f"Predictive Trajectory: {selected_lga}", xaxis_title="Year", yaxis_title="Hospitalizations", 
                      hovermode="x unified", plot_bgcolor='#111111', paper_bgcolor='#111111', font=dict(color='#D4AF37'))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Generative AI Policy Assistant")
    st.markdown(f"The Copilot is analyzing the Random Forest outputs for **{selected_lga}**.")
    
    # Simulate LLM Chat Interface
    prompt = st.chat_input(f"Ask the AI to generate a policy brief for {selected_lga}...")
    
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Synthesizing epidemiological ML data..."):
                time.sleep(1.5) # Simulate AI thinking time
                
                impact = int(baseline_preds[-1] - intervention_preds[-1])
                seifa = current_data['SEIFA_Score']
                
                if funding_boost == 0:
                    st.write(f"**Policy Warning:** Currently, no intervention is planned for {selected_lga}. Based on the Random Forest projections, preventable hospitalizations will reach **{int(baseline_preds[-1])}** by 2030 due to an aging demographic and a SEIFA index of {seifa}. Immediate funding allocation is recommended.")
                else:
                    st.write(f"**Executive Policy Brief: {selected_lga} Intervention**")
                    st.write(f"Based on the input parameters, increasing clinic capacity by **{funding_boost}%** shifts the epidemiological trajectory significantly. Our digital twin model projects that this targeted intervention will avert **{impact}** preventable hospitalizations by 2030.")
                    st.write(f"**Strategic Rationale:** Given {selected_lga}'s SEIFA disadvantage score ({seifa}), allocating these resources not only optimizes total state healthcare expenditure by reducing emergency admissions but strongly aligns with health equity mandates.")
