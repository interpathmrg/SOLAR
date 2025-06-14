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
import matplotlib.pyplot as plt

"""
Calcular_Energia.py
---------------------------------------------
Convierte las predicciones de GHI (generadas por Predict_GHI.py)
 en estimaciones de energía producida por la planta solar SAJOMA.

- Lee un CSV con la columna `GHI_pred`.
- Calcula energía kWh y MWh usando área, eficiencia y duración del intervalo.
- Guarda resultados a CSV y muestra gráfico de producción horaria.
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

"""
Calcular_Energia.py (revisado)
---------------------------------------------
Convierte las predicciones de GHI en energía estimada y genera
una gráfica **siempre**, guardándola como PNG.  Además mostrará
la figura en pantalla cuando el entorno lo permita (no-headless).

Uso básico:
    python calcular_energia_solar.py --plot

Flags:
    --pred   CSV con columna GHI_pred.
    --out    CSV de energía.
    --plot   Activa la generación de la figura (default: False).
    --fig    Ruta del PNG de salida (default result/energia_horaria.png)
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

"""
Calcular_Energia.py (revisado)
---------------------------------------------
Convierte las predicciones de GHI en energía estimada y genera
una gráfica **siempre**, guardándola como PNG.  Además mostrará
la figura en pantalla cuando el entorno lo permita (no-headless).

Uso básico:
    python calcular_energia_solar.py --plot

Flags:
    --pred   CSV con columna GHI_pred.
    --out    CSV de energía.
    --plot   Activa la generación de la figura (default: False).
    --fig    Ruta del PNG de salida (default result/energia_horaria.png)
"""
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

"""
Calcular_Energia.py (optimizado)
---------------------------------------------
• Corrige el typo `def main():():` → `def main():`
• Evita cuelgues al graficar miles de barras: si el archivo tiene
  > 2 000 filas, agrupa por día y grafica la energía diaria.
• Guarda siempre la figura PNG y la muestra solo si el backend lo permite.
"""

AREA_HECTARES = 220
EFFICIENCY = 0.13
INTERVAL_H = 1
AREA_M2 = AREA_HECTARES * 10_000

# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Calcula energía a partir de GHI_pred.")
    p.add_argument("--pred", default="../result/predicciones_GHI.csv",
                   help="CSV con columna GHI_pred.")
    p.add_argument("--out", default="../result/produccion_estimada_SAJOMA.csv",
                   help="CSV de salida con energía estimada.")
    p.add_argument("--plot", action="store_true",
                   help="Generar PNG de la gráfica.")
    p.add_argument("--fig", default="../graph/calcular-energia_horaria.png",
                   help="Ruta del PNG a guardar si --plot está activo.")
    return p.parse_args()

# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_args()

    print("[*] Cargando predicciones desde", args.pred)
    df = pd.read_csv(args.pred, index_col=0, parse_dates=True)

    if "GHI_pred" not in df.columns:
        raise ValueError("El archivo no contiene la columna 'GHI_pred'.")

    # --- Energía ---
    df["Energia_kWh"] = (df["GHI_pred"] * AREA_M2 * EFFICIENCY * INTERVAL_H) / 1000
    df["Energia_MWh"] = df["Energia_kWh"] / 1000

    print(df[["GHI_pred", "Energia_MWh"]].head())

    # --- Gráfica ---
    if args.plot:
        if len(df) > 2000:
            print("[i] Dataset grande → agregando por día para la gráfica.")
            df_plot = df["Energia_MWh"].resample("D").sum()
            xlabel = "Fecha (diario)"
            title = "Producción diaria estimada - Planta Solar Girasol"
            x_vals = df_plot.index
            y_vals = df_plot.values
        else:
            df_plot = df["Energia_MWh"]
            xlabel = "Índice temporal"
            title = "Producción horaria estimada - Planta Solar Girasol"
            x_vals = range(len(df_plot))
            y_vals = df_plot.values

        plt.figure(figsize=(10, 4))
        plt.bar(x_vals, y_vals, color="green")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Energía generada (MWh)")
        plt.grid(axis="y")
        plt.tight_layout()

        # Guardar siempre
        os.makedirs(os.path.dirname(args.fig), exist_ok=True)
        plt.savefig(args.fig, dpi=150)
        print(f"[*] Gráfica guardada en {args.fig}")

        # Mostrar si backend lo permite
        try:
            plt.show()
        except Exception:
            print("[i] Entorno headless: figura guardada, no mostrada en pantalla.")
        plt.close()

    # --- Guardar CSV ---
    df.to_csv(args.out, index=True)
    print("[*] CSV de producción guardado en", args.out)
    print("Listo ✨")

if __name__ == "__main__":
    main()
