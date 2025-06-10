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


## calcular_energia_girasol.py

import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURACIÓN PARA PLANTA GIRASOL ===
area_m2 = 220 * 10000            # 220 hectáreas a m²
eficiencia = 0.13                # eficiencia global estimada
intervalo_horas = 1              # cada fila es una hora

# === CARGAR PREDICCIONES ===
archivo = "/home/mrgonzalez/Desktop/PYTHON/SOLAR/result/predicciones_GHI.csv"
df = pd.read_csv(archivo)

# === CÁLCULO DE ENERGÍA POR HORA ===
df["Energia_kWh"] = (df["Predicted_GHI"] * area_m2 * eficiencia * intervalo_horas) / 1000

# === MOSTRAR RESULTADOS ===
print(df[["Predicted_GHI", "Energia_kWh"]].head())

# Cálculo de energía por hora en kWh y conversión a MW·h
df["Energia_kWh"] = (df["Predicted_GHI"] * area_m2 * eficiencia * intervalo_horas) / 1000
df["Energia_MWh"] = df["Energia_kWh"] / 1000  # conversión a MWh

# === GRÁFICO DE PRODUCCIÓN HORARIA ===
plt.figure(figsize=(10, 5))
plt.bar(range(len(df)), df["Energia_MWh"], color="green")
plt.title("Producción estimada por hora - Planta Solar Girasol")
plt.xlabel("Hora (índice)")
plt.ylabel("Energía generada (MWh)")
plt.grid(axis="y")
plt.tight_layout()
plt.show()


# === GUARDAR RESULTADOS ===
df.to_csv("/home/mrgonzalez/Desktop/PYTHON/SOLAR/result/produccion_estimada_girasol.csv", index=False)
print("Archivo guardado: produccion_estimada_girasol.csv")
