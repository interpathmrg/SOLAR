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
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === CARGA DEL DATASET DESDE PARQUET ===
file_path = "../result/clean_featured.parquet"
df = pd.read_parquet(file_path)

# === SELECCIÓN DE FEATURES ===
features = [
    "temperature", "relative-humidity", "kt", "deltaT",
    "GHI_lag_1h", "GHI_lag_2h", "GHI_lag_3h", "GHI_roll3h",
    "wind-speed", "barometric-preasure", "wind_u", "wind_v",
    "temp_roll3h", "dewpoint",
    "sin_hour", "cos_hour", "sin_doy", "cos_doy"
]
target = "GHI"

X = df[features]
y = df[target]

# === DIVISIÓN TRAIN/TEST TEMPORAL ===
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
r2 = r2_score(y_test, y_pred)
print("[*]")
print(f"MAE: {mae:.2f} es la desviación promedio en W/m² entre la predicción y el valor real.")
print(f"RMSE: {rmse:.2f} penaliza errores grandes, por lo que si hay outliers o picos, RMSE será más alto que MAE.")
print(f"R²: {r2:.3f} ")
print("Interpretación R²:")
print("1.0 → predicción perfecta")
print("0.0 → el modelo no mejora nada sobre el promedio")
print("< 0 → el modelo es peor que simplemente usar el promedio")


# === GRAFICO 1: IMPORTANCIA DE VARIABLES ===
plt.figure(figsize=(10, 5))
sns.barplot(x=model.feature_importances_, y=features)
plt.title("Importancia de las variables (XGBoost)")
plt.xlabel("Importancia")
plt.ylabel("Variable")
plt.tight_layout()
#plt.show()
plt.savefig("../graph/trainner-variables-importancia.png", dpi=150)

# === GRAFICO 2: PREDICCIONES VS VALORES REALES ===
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
plt.xlabel("GHI real")
plt.ylabel("GHI predicho")
plt.title("Dispersión: GHI real vs predicho")
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("../graph/treinner-predvsreal.png", dpi=150)

# === GRAFICO 3: ERROR RESIDUAL ===
errores = y_test - y_pred
plt.figure(figsize=(10, 4))
plt.plot(errores.values, alpha=0.5)
plt.title("Error residual (GHI real - predicho)")
plt.xlabel("Índice")
plt.ylabel("Error (W/m²")
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("../graph/trainner-error-residual.png", dpi=150)

# === GRAFICO 4: ERROR ABSOLUTO MEDIO POR HORA ===
X_test = X_test.copy()
X_test["ErrorAbs"] = np.abs(y_test - y_pred)
# El índice de X_test ya es datetime, tomamos la hora directamente
X_test["Hour"] = X_test.index.hour

errores_por_hora = X_test.groupby("Hour")["ErrorAbs"].mean()

plt.figure(figsize=(10, 4))
errores_por_hora.plot(kind="bar", color="salmon")
plt.title("Error absoluto medio por hora del día")
plt.xlabel("Hora del día")
plt.ylabel("Error absoluto (W/m²)")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("../graph/trainner-error-hora.png", dpi=150)

# === GRAFICO 5: PREDICCIÓN VS TIEMPO ===
plt.figure(figsize=(12, 4))
plt.plot(y_test.values[:500], label="GHI real")
plt.plot(y_pred[:500], label="GHI predicho", alpha=0.7)
plt.title("Comparación GHI real vs. predicho (primeros 500 registros)")
plt.xlabel("Índice temporal")
plt.ylabel("GHI (W/m²)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../graph/trainner-GHIvspredicho.png", dpi=150)

# === GUARDAR MODELO ENTRENADO ===
joblib.dump(model, "../result/modelo_xgboost_GHI.joblib")
print("Modelo guardado como 'modelo_xgboost_GHI.joblib'")
