# https://energyconsumptionpredictor.streamlit.app/

# Energy Consumption (Per Capita) Predictor
Built a machine-learning pipeline that estimates per‑capita energy usage from economic and environmental indicators, applying data‑wrangling, visualization, and Random Forest/XGBoost techniques in AI4ALL’s cutting-edge Ignite accelerator.

# Problem Statement
Reliable energy-usage projections are crucial for policy makers and infrastructure planners, yet many countries lack forward-looking estimates. This project explores historical global data to model future per-capita consumption, informing sustainable development strategies.

# Key Results
Processed 3,649 country-year records spanning 19 energy and economic variables.

Trained a Random Forest Regressor to predict Energy_per_capita_kWh, reporting RMSE and R² metrics after a train/test split.

Generated a 2030 per-capita energy forecast for a hypothetical nation to illustrate model deployment.

Produced feature-importance plots highlighting the most influential drivers of energy consumption.

# Methodologies
Cleaned raw CSV data by removing thousands separators and coercing columns to numeric types.

Dropped rows with missing target values and split data into training and testing subsets.

Fitted a Random Forest Regressor (100 trees, random_state=42), evaluated with MSE, MAE and R², and visualized feature importances via Random Forest and correlation heatmap.

Developed an accompanying Jupyter/Colab notebook for exploratory analysis, correlation studies, model tuning with RandomizedSearchCV, and interactive visualizations using Seaborn and Plotly.

# Data Sources
`data/global_data_on_sustainable_energy.csv`: compiled global energy, economic, and environmental indicators (3,649 rows × 19 columns) used for model training and evaluation sourced from Kaggle at https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy.

# Technologies Used
Python, NumPy, pandas

Scikit-learn

Matplotlib, Seaborn, Plotly

Jupyter Notebook / Google Colab

# Authors
Adrian Morton (amort015@fiu.edu)

Aashreeya Karmacharya (axk6637@mavs.uta.edu)

Darshan Shrestha (dbs6231@mavs.uta.edu)
