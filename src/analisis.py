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
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
file_path = "/home/mrgonzalez/Desktop/PYTHON/SOLAR/data/2017.csv"
try:
    df = pd.read_csv(file_path)
    print("Archivo CSV cargado exitosamente.")

    df = df.drop(columns=['Unnamed: 18'])
    df = df[df["GHI"] > 5]  # descarta valores bajos cerca del amanecer/anochecer
    df["Timestamp"] = pd.to_datetime(df[["Year", "Month", "Day", "Hour", "Minute"]])
    df.set_index("Timestamp", inplace=True)

    # Mostrar las primeras filas para conocer la estructura

    print('[*] Visual de la cabecera y del final del dataset')
    print(df.info)
    print('[*] Describe estadíticamente el dataset')
    print(df.describe())
    print('[*] Registros sin valor')
    print(df.isnull().sum())

    # Promedio diario de GHI
    ghi_daily = df["GHI"].resample("D").mean()
    print(ghi_daily.head())

    # Crear visualizaciones
    plt.figure(figsize=(14, 5))
    plt.plot(ghi_daily, label="GHI diario")
    plt.title("Promedio diario de Irradiancia Global Horizontal (GHI)")
    plt.xlabel("Fecha")
    plt.ylabel("GHI (W/m²)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Boxplot de GHI por hora del día
    df["Hour"] = df.index.hour

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Hour", y="GHI")
    plt.title("Distribución de GHI por hora del día")
    plt.xlabel("Hora del día")
    plt.ylabel("GHI (W/m²)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Matriz de correlación entre variables seleccionadas
    corr_vars = ["GHI", "DNI", "DHI", "Temperature", "Relative Humidity", "Wind Speed", "Pressure"]
    correlation_matrix = df[corr_vars].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Matriz de correlación entre variables meteorológicas y de irradiación")
    plt.tight_layout()
    plt.show()  

except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta: {file_path}")
    df = pd.DataFrame()
except pd.errors.EmptyDataError:
    print(f"Error: El archivo en la ruta: {file_path} esta vacio.")
    df = pd.DataFrame()
except pd.errors.ParserError:
    print(f"Error: No se pudo analizar el archivo CSV en la ruta: {file_path}. Asegúrate de que el archivo tenga el formato correcto.")
    df = pd.DataFrame()
except Exception as e:
    print(f"Error inesperado: {e}")
    df = pd.DataFrame()