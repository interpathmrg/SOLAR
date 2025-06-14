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
import argparse
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# Script: Predict_GHI.py
# Descripción: Carga un modelo XGBoost entrenado previamente y genera predicciones
#              de GHI sobre un archivo Parquet con las mismas features de
#              entrenamiento. Opcionalmente guarda las predicciones a CSV y
#              dibuja una comparación con los valores reales si están presentes.
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Predice GHI usando modelo entrenado XGBoost.")
    parser.add_argument("--data", type=str, default="../result/clean_featured.parquet",
                        help="Ruta al archivo Parquet de entrada con las features.")
    parser.add_argument("--model", type=str, default="../result/modelo_xgboost_GHI.joblib",
                        help="Ruta al modelo .joblib entrenado.")
    parser.add_argument("--output", type=str, default="../result/predicciones_GHI.csv",
                        help="Ruta de salida CSV con las predicciones de GHI.")
    parser.add_argument("--plot", action="store_true", help="Mostrar gráficas de comparación.")
    return parser.parse_args()


def main():
    args = parse_args()

    # === 1. Cargar modelo ===
    print("[*] Cargando modelo desde", args.model)
    model = joblib.load(args.model)

    # === 2. Cargar datos ===
    print("[*] Cargando datos desde", args.data)
    df = pd.read_parquet(args.data)

    # === 3. Seleccionar features en el mismo orden que el entrenamiento ===
    features = [
        "temperature", "relative-humidity", "kt", "deltaT",
        "GHI_lag_1h", "GHI_lag_2h", "GHI_lag_3h", "GHI_roll3h",
        "wind-speed", "barometric-preasure", "wind_u", "wind_v",
        "temp_roll3h", "dewpoint",
        "sin_hour", "cos_hour", "sin_doy", "cos_doy"
    ]

    X = df[features]

    # === 4. Generar predicciones ===
    print("[*] Generando predicciones …")
    df["GHI_pred"] = model.predict(X)

    # === 5. Guardar predicciones ===
    df[["GHI_pred"]].to_csv(args.output, index=True)
    print(f"[*] Predicciones guardadas en {args.output}")

    # === 6. Gráficas opcionales ===
    if args.plot and "GHI" in df.columns:
        y_true = df["GHI"]
        y_pred = df["GHI_pred"]

        # Dispersión real vs predicho
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.3)
        plt.plot([0, y_true.max()], [0, y_true.max()], "r--")
        plt.xlabel("GHI real")
        plt.ylabel("GHI predicho")
        plt.title("Dispersión: GHI real vs predicho")
        plt.grid(True)
        plt.tight_layout()
        #plt.show()
        plt.savefig("../graph/produccion-dispersion.png", dpi=150)

        # Serie temporal de los primeros 500 registros
        plt.figure(figsize=(12, 4))
        plt.plot(y_true.values[:500], label="GHI real")
        plt.plot(y_pred.values[:500], label="GHI predicho", alpha=0.7)
        plt.title("Comparación GHI real vs. predicho (primeros 500 registros)")
        plt.xlabel("Índice temporal")
        plt.ylabel("GHI (W/m²)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        #plt.show()
        plt.savefig("../graph/produccion-realvspredicho", dpi=150)

if __name__ == "__main__":
    main()
