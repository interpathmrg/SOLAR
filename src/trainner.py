"""
Desarrollado por Miguel Gonzalez
Asistido por  ChatGPT
10-Jun-2025
Análsisis y predicción de un dataset :
https://www.kaggle.com/datasets/ibrahimkiziloklu/solar-radiation-dataset

Analisis, Entrenamiento del modelo y predicción del promedio
de la irradiación total GHI y el calculo de la energía producida
por el parque.
-------------------------------------------------------------------------------
"""

# modelo_predictivo.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === CARGA Y PREPROCESAMIENTO ===
file_path = "/home/mrgonzalez/Desktop/PYTHON/SOLAR/data/2017.csv"
df = pd.read_csv(file_path)

df = df.drop(columns=["Unnamed: 18"], errors="ignore")
df = df[df["GHI"] > 5]  # descarta valores bajos cerca del amanecer/anochecer
df["Timestamp"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
df.set_index("Timestamp", inplace=True)

# Eliminar filas sin GHI
df = df.dropna(subset=["GHI"])

# Selección de características
features = ["Temperature", "Relative Humidity", "DNI", "DHI", "Wind Speed", "Pressure"]
target = "GHI"

X = df[features]
y = df[target]

# División temporal en train/test
split_index = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# === ENTRENAMIENTO ===
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# === EVALUACIÓN ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# === GRAFICO 1: IMPORTANCIA DE VARIABLES ===
plt.figure(figsize=(10, 5))
sns.barplot(x=model.feature_importances_, y=features)
plt.title("Importancia de las variables (XGBoost)")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()

# === GRAFICO 2: PREDICCIONES VS VALORES REALES ===
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
plt.xlabel("GHI real")
plt.ylabel("GHI predicho")
plt.title("Dispersión: GHI real vs predicho")
plt.grid(True)
plt.tight_layout()
plt.show()



# === GRAFICO 3: ERROR RESIDUAL ===
errores = y_test - y_pred
plt.figure(figsize=(10, 4))
plt.plot(errores.values, alpha=0.5)
plt.title("Error residual (GHI real - predicho)")
plt.xlabel("Índice")
plt.ylabel("Error (W/m²)")
plt.grid(True)
plt.tight_layout()
plt.show()


# === MÉTRICA ADICIONAL: R² ===
r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.3f}")

# === GRAFICO 4: Error absoluto promedio por hora del día ===
X_test["ErrorAbs"] = np.abs(y_test - y_pred)
X_test["Hour"] = X_test.index.hour
errores_por_hora = X_test.groupby("Hour")["ErrorAbs"].mean()

plt.figure(figsize=(10, 4))
errores_por_hora.plot(kind="bar", color="salmon")
plt.title("Error absoluto medio por hora del día")
plt.xlabel("Hora del día")
plt.ylabel("Error absoluto (W/m²)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# === GUARDAR MODELO ===
joblib.dump(model, "/home/mrgonzalez/Desktop/PYTHON/SOLAR/result/modelo_xgboost_GHI.joblib")
print("Modelo guardado como 'modelo_xgboost_GHI.joblib'")
