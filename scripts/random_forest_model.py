import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset from the data directory
df = pd.read_csv('data/raw/global_data_on_sustainable_energy.csv')

# Remove thousands separators and coerce numeric values
df = df.replace(',', '', regex=True)
df = df.apply(pd.to_numeric, errors='ignore')

# Drop rows where the target is missing
df_rf = df.dropna(subset=['Energy_per_capita_kWh'])

# Define features and target
X = df_rf.drop(columns=['Country', 'Year', 'Energy_per_capita_kWh'])
y = df_rf['Energy_per_capita_kWh']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Random Forest Regression Results")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Simulated statistics for a hypothetical future nation
future_values = [
    16000,   # GDP_per_capita
    1.2,     # Energy_intensity_MJ_per_GDP
    20.5,    # Renewables_equivalent_pct
    85.0,    # Clean_fuels_pct
    150,     # Density_P_Km2
    700000,  # Land_area_Km2
    2.5,     # Renewable_capacity_per_capita
    30.0,    # Renewable_share_pct
    100000,  # CO2_kt
    250,     # Electricity_fossil_TWh
    60.0,    # Low_carbon_electricity_pct
    150,     # Electricity_renewable_TWh
    2.5,     # GDP_growth
    10,      # Electricity_nuclear_TWh
    95.0,    # Access_electricity_pct
    1e8      # Energy_financial_flows_usd
]

# Create DataFrame of simulated statistics
future_data = pd.DataFrame([future_values], columns=X.columns)

# Predict future energy consumption
future_prediction = rf.predict(future_data)
print(f"\nEstimated energy use per capita in 2030: {future_prediction[0]:.2f} kWh")

importances = rf.feature_importances_
feature_names = X.columns
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importances from Random Forest")
plt.bar(range(len(importances)), importances[sorted_indices])
plt.xticks(range(len(importances)), feature_names[sorted_indices], rotation=90)
plt.tight_layout()
plt.show()
