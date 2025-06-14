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
Features:
timestamp
wind-speed
wind-dir
barometric-preasure
temperature
relative-humidity
GHI
GHI-back
precipitation // no tiene datos

-------------------------------------------------------------------------------

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from pvlib.location import Location

# ------------------------------
# Función para aplicar Isolation Forest
# ------------------------------
def aplicar_isolation_forest(df, features, contamination=0.01):
    df = df.dropna(subset=features)
    iso = IsolationForest(contamination=contamination, random_state=0)
    mask = iso.fit_predict(df[features]) == 1
    df["if_score"] = iso.decision_function(df[features])
    df["if_outlier"] = ~mask

    print("[*] Conteo de outliers detectados:")
    print(df["if_outlier"].value_counts())

    # Gráfica auxiliar: presión vs temperatura
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df, x="barometric-preasure", y="temperature",
        hue="if_outlier", palette={False: "blue", True: "red"}, alpha=0.5
    )
    plt.title("Detección de Outliers por Isolation Forest")
    plt.xlabel("Presión Barométrica (hPa)")
    plt.ylabel("Temperatura (°C)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend(title="¿Outlier?")
    #plt.show()
    plt.savefig("../graph/analisis-outliners.png", dpi=150)

    return df[mask]

# ------------------------------
# Inicio del flujo principal
# ------------------------------

file_path = r"../data/SAJOMA-Solar-2025.csv"
try:
    df = pd.read_csv(file_path)
    print("[*] Archivo cargado exitosamente.")

    # --- Limpieza inicial ---
    df = df.drop(columns=['precipitation'])
    df = df[df["GHI"] > 5]
    df = df[df["GHI"] < 900]

    # Límites físicos
    phys_limits = (
        (df["barometric-preasure"].between(870, 1040)) &
        (df["temperature"].between(15, 45))
    )
    df = df[phys_limits]

    # IQR univariante
    for col in ["barometric-preasure", "temperature"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df = df[df[col].between(lower, upper)]

    # --- Preparar timestamp e index temporal ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M", errors="raise")
    df.set_index("timestamp", inplace=True)

    # --- Feature engineering ---
    df["hour"] = df.index.hour
    df["dayofyear"] = df.index.dayofyear
    df["sin_hour"] = np.sin(2*np.pi*df["hour"] / 24)
    df["cos_hour"] = np.cos(2*np.pi*df["hour"] / 24)
    df["sin_doy"] = np.sin(2*np.pi*df["dayofyear"] / 365)
    df["cos_doy"] = np.cos(2*np.pi*df["dayofyear"] / 365)

    theta = np.deg2rad(df["wind-dir"])
    df["wind_u"] = df["wind-speed"] * np.sin(theta)
    df["wind_v"] = df["wind-speed"] * np.cos(theta)

    for h in [1, 2, 3]:
        df[f"GHI_lag_{h}h"] = df["GHI"].shift(h)

    df["GHI_roll3h"] = df["GHI"].rolling(window=3).mean()
    df["temp_roll3h"] = df["temperature"].rolling(window=3).mean()

    loc = Location(latitude=18.3352, longitude=-70.19596, tz='America/Santo_Domingo')
    clear_sky = loc.get_clearsky(df.index)["ghi"]
    df["kt"] = df["GHI"] / clear_sky.clip(lower=1)

    df["dewpoint"] = df.apply(lambda r: r["temperature"] - ((100 - r["relative-humidity"]) / 5), axis=1)
    df["deltaT"] = df["temperature"] - df["dewpoint"]

    # --- Aplicar Isolation Forest con features extendidas ---
    features_for_if = [
        "GHI", "temperature", "relative-humidity", "kt",
        "deltaT", "GHI_lag_1h", "GHI_roll3h"
    ]
    df = aplicar_isolation_forest(df, features_for_if)

    # --- Visualizaciones y resumen ---
    print("[*] Describe estadísticamente el dataset")
    print(df.describe())
    print("[*] Registros sin valor")
    print(df.isnull().sum())

    ghi_daily = df["GHI"].resample("D").mean()
    plt.figure(figsize=(14, 5))
    plt.plot(ghi_daily, label="GHI diario")
    plt.title("Promedio diario de Irradiancia Global Horizontal (GHI)")
    plt.xlabel("Fecha")
    plt.ylabel("GHI (W/m²)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig("../graph/analisis-timeserie-GHI.png", dpi=150)

    df["Hour"] = df.index.hour
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="Hour", y="GHI")
    plt.title("Distribución de GHI por hora del día")
    plt.xlabel("Hora del día")
    plt.ylabel("GHI (W/m²)")
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig("../graph/analisis-GHI-hora.png", dpi=150)

    corr_vars = features_for_if + [
        "wind-speed", "barometric-preasure", "wind-dir",
        "hour", "dayofyear", "sin_hour", "cos_hour",
        "sin_doy", "cos_doy", "wind_u", "wind_v",
        "temp_roll3h", "dewpoint"
    ]
    correlation_matrix = df[corr_vars].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Matriz de correlación entre variables meteorológicas y de irradiación")
    plt.tight_layout()
    #plt.show()
    plt.savefig("../graph/analisis-matriz-correlacion.png", dpi=150)

    df.to_parquet(r"../result/clean_featured.parquet")
    print("[*] Archivo parquet generado correctamente.")

except FileNotFoundError:
    print(f"[!] Error: No se encontró el archivo en la ruta: {file_path}")
    df = pd.DataFrame()
except pd.errors.EmptyDataError:
    print(f"[!] Error: El archivo en la ruta: {file_path} esta vacio.")
    df = pd.DataFrame()
except pd.errors.ParserError:
    print(f"[!] Error: No se pudo analizar el archivo CSV en la ruta: {file_path}.")
    df = pd.DataFrame()
except Exception as e:
    print(f"[!] Error inesperado: {e}")
    df = pd.DataFrame()
