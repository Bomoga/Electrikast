import streamlit as st
from pathlib import Path
import plotly.graph_objects as go
import pandas as pd
import joblib
import numpy as np
import os
from predict import EnergyPredictor


# -------------------------------------------------
# Define a fixed project root (two levels up from this file)
# src/app.py  -> repo root is parents[1]
ROOT = Path(__file__).resolve().parents[1]

# Filepaths
MODEL_PATH = ROOT / "models" / "xgboost_model.pkl"
DATA_PATH  = ROOT / "data" / "global-data-on-sustainable-energy.csv"
FIGURE_PATH = ROOT / "models" / "figures" / "globalenergyimage.jpg"
FEATURE_ORDER_PATH = ROOT / "data" / "feature_order.json"

# Initialize model
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

# Initialize dataset
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Data not found at: {DATA_PATH}")
        st.stop()
    return pd.read_csv(DATA_PATH)

# Load once at startup
model = load_model()
df = load_data()

predictor = EnergyPredictor(MODEL_PATH, FEATURE_ORDER_PATH)

# Display overview image and title
if os.path.exists(FIGURE_PATH):
    st.image(FIGURE_PATH, use_container_width=True)
st.title("Electrikast: Energy Forecaster")

# Link to notebook hosted on Colab
colab_url = "https://colab.research.google.com/github/axk6637/ai4all/blob/main/EnergyUsage.ipynb#scrollTo=4slJU_rhuunD"
if st.button("Check out our work!"):
    st.markdown(f"[Check out our work!]({colab_url})", unsafe_allow_html=True)

st.markdown("""
### How to use this forecaster:
1. Select the **Energy Forecast** tab.
2. Enter the scenario values (electricity mix, GDP, CO₂, etc.).
3. Click **Forecast** to generate the per-capita energy estimate.
4. Tweak inputs to compare scenarios. Use **Global Energy Graph** for map-based exploration.

**What the result means:**
- **Forecast** means the predicted *primary energy consumption per person (kWh/person)*.

**Having issues?**
- Try refreshing. If errors persist, the model or data may be missing from the deploy.
""")

# Tabs for prediction and visualization
tab1, tab2 = st.tabs(["Energy Forecaster", "Global Energy Graph"])

with tab1:
    st.header("Feature Input")
    access_electricity_pct = st.number_input(
        "Access to Electricity (% of population)",
        min_value=0.0, max_value=100.0, value=95.0, key="access_electricity_pct"
    )

    access_clean_fuels = st.number_input(
        "Access to Clean Fuels for Cooking (% of population)",
        min_value=0.0, max_value=100.0, value=80.0, key="access_clean_fuels"
    )

    renewable_capacity_per_capita = st.number_input(
        "Renewable Electricity Capacity per Person (kW per capita)",
        min_value=0.0, value=2.5, key="renewable_capacity_per_capita"
    )

    renewable_share_pct = st.number_input(
        "Renewable Energy Share in Final Consumption (%)",
        min_value=0.0, max_value=100.0, value=30.0, key="renewable_share_pct"
    )

    electricity_fossil_twh = st.number_input(
        "Electricity Generated from Fossil Fuels (TWh)",
        min_value=0.0, value=250.0, key="electricity_fossil_twh"
    )

    electricity_nuclear_twh = st.number_input(
        "Electricity Generated from Nuclear (TWh)",
        min_value=0.0, value=10.0, key="electricity_nuclear_twh"
    )

    electricity_renewable_twh = st.number_input(
        "Electricity Generated from Renewables (TWh)",
        min_value=0.0, value=150.0, key="electricity_renewable_twh"
    )

    low_carbon_electricity_pct = st.number_input(
        "Low-Carbon Electricity (% of total electricity)",
        min_value=0.0, max_value=100.0, value=60.0, key="low_carbon_electricity_pct"
    )

    energy_intensity = st.number_input(
        "Energy Intensity (MJ per $2017 PPP GDP)",
        min_value=0.0, value=1.2, key="energy_intensity"
    )

    co2_kt = st.number_input(
        "CO₂ Emissions (kilotonnes per year)",
        min_value=0.0, value=100000.0, key="co2_kt"
    )

    renewables_equiv_primary_energy = st.number_input(
        "Renewables Share of Primary Energy (%)",
        min_value=0.0, max_value=100.0, value=20.0, key="renewables_equiv_primary_energy"
    )

    gdp_growth = st.number_input(
        "GDP Growth Rate (%)",
        min_value=0.0, max_value=100.0, value=2.5, key="gdp_growth"
    )

    gdp_per_capita = st.number_input(
        "GDP per Capita (USD, PPP adjusted)",
        min_value=0.0, value=16000.0, key="gdp_per_capita"
    )

    density_p_km2 = st.number_input(
        "Population Density (people per km²)",
        min_value=0.0, value=150.0, key="density_p_km2"
    )

    land_area_km2 = st.number_input(
        "Land Area (km²)",
        min_value=0.0, value=700000.0, key="land_area_km2"
    )

    latitude = st.number_input(
        "Latitude (degrees)",
        min_value=-90.0, max_value=90.0, value=0.0, key="latitude"
    )

    longitude = st.number_input(
        "Longitude (degrees)",
        min_value=-180.0, max_value=180.0, value=0.0, key="longitude"
    )

    if st.button("Forecast"):
        inputs = {
            "access_electricity_pct": access_electricity_pct,
            "access_clean_fuels": access_clean_fuels,
            "renewable_capacity_per_capita": renewable_capacity_per_capita,
            "renewable_share_pct": renewable_share_pct,
            "electricity_fossil_twh": electricity_fossil_twh,
            "electricity_nuclear_twh": electricity_nuclear_twh,
            "electricity_renewable_twh": electricity_renewable_twh,
            "low_carbon_electricity_pct": low_carbon_electricity_pct,
            "energy_intensity": energy_intensity,
            "co2_kt": co2_kt,
            "renewables_equiv_primary_energy": renewables_equiv_primary_energy,
            "gdp_growth": gdp_growth,
            "gdp_per_capita": gdp_per_capita,
            "density_p_km2": density_p_km2,
            "land_area_km2": land_area_km2,
            "latitude": latitude,
            "longitude": longitude,
        }
        
        predictor.validate(inputs)
        prediction = predictor.predict(inputs)
        st.success(f"XGBoost Prediction: {prediction:.2f} kWh per capita.")

with tab2:
    st.header("Global Energy Map")
    column_name = st.selectbox("Select column to visualize", [
        'Primary energy consumption per capita (kWh/person)',
        'Access to electricity (% of population)',
        'Renewable energy share in the total final energy consumption (%)',
        'Low-carbon electricity (% electricity)',
        'Value_co2_emissions_kt_by_country',
        'Renewables (% equivalent primary energy)',
        'gdp_growth',
        'gdp_per_capita',
    ])
    years = sorted(df['Year'].unique())
    selected_year = st.slider("Select Year", min_value=int(years[0]), max_value=int(years[-1]), value=int(years[0]))
    filtered_df = df[df['Year'] == selected_year]

    fig = go.Figure(go.Choropleth(
        locations=filtered_df['Entity'],
        z=filtered_df[column_name],
        locationmode='country names',
        colorscale='Cividis',
        colorbar=dict(title=column_name),
        zmin=df[column_name].min(),
        zmax=df[column_name].max(),
    ))
    fig.update_layout(
        title_text=f'{column_name} Map - {selected_year}',
        geo=dict(showframe=True, showcoastlines=True, projection_type='natural earth'),
        height=400,
        width=800,
        font=dict(family='Arial', size=12),
        margin=dict(t=80, l=50, r=50, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)
