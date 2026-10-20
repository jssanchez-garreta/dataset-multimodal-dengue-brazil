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

# ============================================================
# AUDITORÍA PANEL DE ENTRADA
# ============================================================

print("\n===================================")
print("INPUT PANEL AUDIT")
print("===================================")

duplicados_panel = (
    df
    .groupby(["codigo_ibge", "SE"])
    .size()
    .gt(1)
    .sum()
)

max_rep_panel = (
    df
    .groupby(["codigo_ibge", "SE"])
    .size()
    .max()
)

print(
    f"Duplicated municipality-week keys: "
    f"{duplicados_panel}"
)

print(
    f"Maximum repetitions per key: "
    f"{max_rep_panel}"
)

if duplicados_panel > 0:
    raise ValueError(
        "Duplicated municipality-week keys detected "
        "before temporal feature generation."
    )

# ============================================================
# AUDITORÍA CONTINUIDAD TEMPORAL
# ============================================================

print("\n===================================")
print("TEMPORAL CONTINUITY AUDIT")
print("===================================")

df["data_iniSE"] = pd.to_datetime(
    df["data_iniSE"]
)

municipios_con_saltos = 0
saltos_totales = 0

for codigo_ibge, grupo in df.groupby("codigo_ibge"):

    grupo = grupo.sort_values("data_iniSE")

    diff_days = (
        grupo["data_iniSE"]
        .diff()
        .dt.days
        .dropna()
    )

    saltos = (
        diff_days != 7
    ).sum()

    if saltos > 0:

        municipios_con_saltos += 1
        saltos_totales += saltos

print(
    f"Municipalities with temporal gaps: "
    f"{municipios_con_saltos}"
)

print(
    f"Total temporal gaps: "
    f"{saltos_totales}"
)

if municipios_con_saltos > 0:

    print(
        f"WARNING: {municipios_con_saltos} municipalities "
        f"contain temporal gaps."
    )

if municipios_con_saltos == 0:

    print(
        "OK: temporal continuity validated."
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
        "data_iniSE"
    ]
)

df = df.reset_index(drop=True)

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

print("\n===================================")
print("LAG AUDIT")
print("===================================")

for col in temporal_features:

    print(
        f"{col}: "
        f"{df[col].isna().sum():,} missing"
    )

# ============================================================
# AUDITORÍA PANEL FINAL
# ============================================================

print("\n===================================")
print("OUTPUT PANEL AUDIT")
print("===================================")

duplicados_panel = (
    df
    .groupby(["codigo_ibge", "SE"])
    .size()
    .gt(1)
    .sum()
)

max_rep_panel = (
    df
    .groupby(["codigo_ibge", "SE"])
    .size()
    .max()
)

print(
    f"Duplicated municipality-week keys: "
    f"{duplicados_panel}"
)

print(
    f"Maximum repetitions per key: "
    f"{max_rep_panel}"
)

if duplicados_panel > 0:
    raise ValueError(
        "Temporal feature generation introduced "
        "duplicated municipality-week keys."
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