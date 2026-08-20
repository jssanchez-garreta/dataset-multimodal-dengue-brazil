# ============================================================
# 24_build_temporal_lags.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Crear variables temporales derivadas:
#
# - Lags epidemiológicos
# - Lags climáticos
# - Ventanas móviles
#
# ENTRADA:
#
# dataset_multimodal_v7.parquet
#
# SALIDA:
#
# dataset_multimodal_v8.parquet
#
# ============================================================

# ==================================================
# IMPORTS
# ==================================================

import os
import time

import pandas as pd

from google.colab import drive

# ==================================================
# GOOGLE DRIVE
# ==================================================

if not os.path.ismount("/content/drive"):
    drive.mount("/content/drive")

# ==================================================
# TIMER
# ==================================================

start_time = time.time()

# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL"
)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "DATASETS_MASTER_MULTIMODAL"
)

INPUT_FILE = os.path.join(
    DATASET_DIR,
    "dataset_multimodal_v7.parquet"
)

OUTPUT_FILE = os.path.join(
    DATASET_DIR,
    "dataset_multimodal_v8.parquet"
)

# ==================================================
# LOAD DATASET
# ==================================================

print("\n===================================")
print("LOADING DATASET V7")
print("===================================")

df = pd.read_parquet(INPUT_FILE)

print(
    f"Shape original: {df.shape}"
)

# ==================================================
# SORT TEMPORAL
# ==================================================

print("\n===================================")
print("SORTING")
print("===================================")

df = df.sort_values(
    [
        "codigo_ibge",
        "anio",
        "semana"
    ]
)

# ==================================================
# CREAR LAGS
# ==================================================

print("\n===================================")
print("CREATING LAGS")
print("===================================")

LAGS = {

    "casos": [1, 2, 4, 8],

    "Rt": [1, 2, 4],

    "precip_total_semana": [1, 4],

    "tempmed": [1, 4],

    "umidmed": [1, 4]

}

for variable, lag_list in LAGS.items():

    print(f"\nVariable: {variable}")

    for lag in lag_list:

        new_col = (
            f"{variable}_lag_{lag}"
        )

        print(
            f"  -> {new_col}"
        )

        df[new_col] = (
            df.groupby(
                "codigo_ibge"
            )[variable]
            .shift(lag)
        )

# ==================================================
# ROLLING WINDOWS
# ==================================================

print("\n===================================")
print("ROLLING WINDOWS")
print("===================================")

# --------------------------------------------------
# CASOS
# --------------------------------------------------

print("casos rolling")

df["casos_roll4_mean"] = (

    df.groupby(
        "codigo_ibge"
    )["casos"]

    .transform(

        lambda x:

        x.shift(1)

        .rolling(
            window=4,
            min_periods=1
        )

        .mean()

    )

)

df["casos_roll8_mean"] = (

    df.groupby(
        "codigo_ibge"
    )["casos"]

    .transform(

        lambda x:

        x.shift(1)

        .rolling(
            window=8,
            min_periods=1
        )

        .mean()

    )

)

# --------------------------------------------------
# PRECIPITACIÓN
# --------------------------------------------------

print("precip rolling")

df["precip_roll4_sum"] = (

    df.groupby(
        "codigo_ibge"
    )["precip_total_semana"]

    .transform(

        lambda x:

        x.shift(1)

        .rolling(
            window=4,
            min_periods=1
        )

        .sum()

    )

)

df["precip_roll8_sum"] = (

    df.groupby(
        "codigo_ibge"
    )["precip_total_semana"]

    .transform(

        lambda x:

        x.shift(1)

        .rolling(
            window=8,
            min_periods=1
        )

        .sum()

    )

)

# --------------------------------------------------
# TEMPERATURA
# --------------------------------------------------

print("temp rolling")

df["tempmed_roll4_mean"] = (

    df.groupby(
        "codigo_ibge"
    )["tempmed"]

    .transform(

        lambda x:

        x.shift(1)

        .rolling(
            window=4,
            min_periods=1
        )

        .mean()

    )

)

df["tempmed_roll8_mean"] = (

    df.groupby(
        "codigo_ibge"
    )["tempmed"]

    .transform(

        lambda x:

        x.shift(1)

        .rolling(
            window=8,
            min_periods=1
        )

        .mean()

    )

)

# ==================================================
# VALIDACIÓN
# ==================================================

print("\n===================================")
print("VALIDATION")
print("===================================")

temporal_features = [

    c

    for c in df.columns

    if (
        "_lag_" in c
        or
        "_roll" in c
    )
]

print(
    f"Nuevas variables: "
    f"{len(temporal_features)}"
)

print("\nLISTADO")

for c in sorted(temporal_features):

    print(c)

print(
    f"\nShape final: "
    f"{df.shape}"
)

# ==================================================
# EXPORT
# ==================================================

print("\n===================================")
print("EXPORTING PARQUET")
print("===================================")

df.to_parquet(
    OUTPUT_FILE,
    index=False
)

# ==================================================
# TIEMPO
# ==================================================

elapsed = (
    time.time()
    - start_time
)

print(
    f"\nTiempo total: "
    f"{elapsed/60:.2f} minutos"
)

print("\nFINALIZADO")