import pandas as pd
import numpy as np

# ------------- Configuración -----------------
SOURCE_PARQUET = "../result/clean_featured.parquet"
OUTPUT_PARQUET = "../result/synthetic_features.parquet"
N_SYNTH = 5000          # filas sintéticas a generar
FEATURES = [
    "temperature", "relative-humidity", "kt", "deltaT",
    "GHI_lag_1h", "GHI_lag_2h", "GHI_lag_3h", "GHI_roll3h",
    "wind-speed", "barometric-preasure", "wind_u", "wind_v",
    "temp_roll3h", "dewpoint",
    "sin_hour", "cos_hour", "sin_doy", "cos_doy"
]

# ------------- 1. Leer dataset limpio --------
df = pd.read_parquet(SOURCE_PARQUET)

# ------------- 2. Preparar stats -------------
stats = df[FEATURES].agg(["mean", "std"])  # media y sigma por columna

# ------------- 3. Generar datos sintéticos ---
rng = np.random.default_rng(42)
synthetic = pd.DataFrame({
    col: rng.normal(loc=stats.loc["mean", col],
                    scale=stats.loc["std",  col],
                    size=N_SYNTH)
    for col in FEATURES
})

# Respeta límites físicos en algunas variables intensas
synthetic["wind-speed"]        = synthetic["wind-speed"].clip(lower=0)
synthetic["relative-humidity"] = synthetic["relative-humidity"].clip(0, 100)
synthetic["kt"]                = synthetic["kt"].clip(lower=0)

# Si quieres que las variables angulares queden en rango [-1, 1]
for col in ["sin_hour", "cos_hour", "sin_doy", "cos_doy"]:
    synthetic[col] = synthetic[col].clip(-1, 1)

# ------------- 4. Guardar parquet ------------
synthetic.to_parquet(OUTPUT_PARQUET)
print(f"Sintético guardado en {OUTPUT_PARQUET} con shape {synthetic.shape}")
