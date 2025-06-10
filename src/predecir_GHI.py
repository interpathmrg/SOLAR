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


# predecir_GHI.py

import pandas as pd
import joblib

# Cargar modelo entrenado
modelo = joblib.load("/home/mrgonzalez/Desktop/PYTHON/SOLAR/result/modelo_xgboost_GHI.joblib")

# Cargar nuevo CSV con variables meteorológicas
nuevo_csv = "/home/mrgonzalez/Desktop/PYTHON/SOLAR/data/clima_futuro.csv"  # <-- cambia esto a tu archivo real
df_nuevo = pd.read_csv(nuevo_csv)

# Preprocesar (asegurar que tenga las columnas necesarias)
features = ["Temperature", "Relative Humidity", "DNI", "DHI", "Wind Speed", "Pressure"]
if all(col in df_nuevo.columns for col in features):
    predicciones = modelo.predict(df_nuevo[features])
    df_nuevo["Predicted_GHI"] = predicciones
    print(df_nuevo[["Predicted_GHI"]].head())

    # Guardar con predicciones
    df_nuevo.to_csv("/home/mrgonzalez/Desktop/PYTHON/SOLAR/result/predicciones_GHI.csv", index=False)
    print("Predicciones guardadas en predicciones_GHI.csv")
else:
    print("Error: el archivo no contiene todas las columnas necesarias.")
