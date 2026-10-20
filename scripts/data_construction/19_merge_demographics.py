# ============================================================
# 19_merge_demographics.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Incorporar variables demográficas
# (IBGE Censo 2022 / SIDRA)
# al dataset multimodal.
#
# ENTRADAS:
#
# dataset_multimodal_v4.parquet
# demographics_brasil.parquet
#
# SALIDAS:
#
# dataset_multimodal_v5.csv
# dataset_multimodal_v5.parquet
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
# CRONÓMETRO
# ==================================================

start_time = time.time()

# ==================================================
# DIRECTORIOS
# ==================================================

PROJECT_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL"
)

DATASETS_DIR = os.path.join(
    PROJECT_DIR,
    "DATASETS_MASTER_MULTIMODAL"
)

DEMOGRAPHICS_DIR = os.path.join(
    PROJECT_DIR,
    "DEMOGRAPHICS_BRASIL"
)

# ==================================================
# ARCHIVOS
# ==================================================

INPUT_DATASET = os.path.join(
    DATASETS_DIR,
    "dataset_multimodal_v4.parquet"
)

DEMOGRAPHICS_FILE = os.path.join(
    DEMOGRAPHICS_DIR,
    "demographics_brasil.parquet"
)

OUTPUT_PARQUET = os.path.join(
    DATASETS_DIR,
    "dataset_multimodal_v5.parquet"
)

OUTPUT_CSV = os.path.join(
    DATASETS_DIR,
    "dataset_multimodal_v5.csv"
)

# ==================================================
# CARGAR DATASET MULTIMODAL
# ==================================================

print("\n===================================")
print("LEYENDO DATASET MULTIMODAL")
print("===================================")

df = pd.read_parquet(
    INPUT_DATASET
)

df["codigo_ibge"] = (
    df["codigo_ibge"]
    .astype(str)
)

print(
    f"Filas: {len(df):,}"
)

print(
    f"Municipios: "
    f"{df['codigo_ibge'].nunique():,}"
)

# ==================================================
# AUDITORÍA PANEL DE ENTRADA
# ==================================================

print("\n===================================")
print("AUDITORÍA PANEL DE ENTRADA")
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
        "before merge."
    )

# ==================================================
# CARGAR DEMOGRAPHICS
# ==================================================

print("\n===================================")
print("LEYENDO DEMOGRAPHICS")
print("===================================")

demographics = pd.read_parquet(
    DEMOGRAPHICS_FILE
)

demographics["codigo_ibge"] = (
    demographics["codigo_ibge"]
    .astype(str)
)

print(
    f"Municipios: "
    f"{len(demographics):,}"
)

# ==================================================
# VARIABLES DEMOGRÁFICAS
# ==================================================

demographics_cols = [
    "codigo_ibge",
    "population_total",
    "male_population_pct",
    "female_population_pct",
    "aging_index",
    "median_age",
    "sex_ratio"
]

demographics = (
    demographics[
        demographics_cols
    ]
)

# ==================================================
# VALIDACIÓN PRE-MERGE
# ==================================================

print("\n===================================")
print("VALIDACIÓN PRE-MERGE")
print("===================================")

duplicados = (
    demographics
    .duplicated(
        subset=["codigo_ibge"]
    )
    .sum()
)

print(
    f"Duplicados demographics: "
    f"{duplicados}"
)

if duplicados > 0:
    raise ValueError(
        f"Demographics table contains "
        f"{duplicados} duplicated municipality IDs."
    )

# ==================================================
# MERGE
# ==================================================

print("\n===================================")
print("MERGE")
print("===================================")

df = df.merge(
    demographics,
    on="codigo_ibge",
    how="left"
)

# ==================================================
# VALIDACIÓN POST-MERGE
# ==================================================

print("\n===================================")
print("VALIDACIÓN POST-MERGE")
print("===================================")

print(
    f"Filas dataset final: "
    f"{len(df):,}"
)

print(
    f"Municipios únicos: "
    f"{df['codigo_ibge'].nunique():,}"
)

duplicados_postmerge = (
    df
    .groupby(["codigo_ibge", "SE"])
    .size()
    .gt(1)
    .sum()
)

max_rep_postmerge = (
    df
    .groupby(["codigo_ibge", "SE"])
    .size()
    .max()
)

print(
    f"Duplicated municipality-week keys: "
    f"{duplicados_postmerge}"
)

print(
    f"Maximum repetitions per key: "
    f"{max_rep_postmerge}"
)

if duplicados_postmerge > 0:
    raise ValueError(
        "Merge introduced duplicated "
        "municipality-week keys."
    )

nulos_demographics = (
    df["population_total"]
    .isna()
    .sum()
)

print(
    f"Nulos demographics: "
    f"{nulos_demographics:,}"
)

municipios_sin_demographics = (
    df.loc[
        df["population_total"].isna(),
        "codigo_ibge"
    ]
    .nunique()
)

print(
    f"Municipios sin demographics: "
    f"{municipios_sin_demographics}"
)

# ==================================================
# VALIDACIÓN VARIABLES
# ==================================================

print("\n===================================")
print("VALIDACIÓN VARIABLES")
print("===================================")

validation_cols = [
    "population_total",
    "male_population_pct",
    "female_population_pct",
    "aging_index",
    "median_age",
    "sex_ratio"
]

print(
    df[
        validation_cols
    ]
    .describe()
)

validacion_panel = pd.DataFrame({
    "metric": [
        "rows",
        "municipalities",
        "duplicated_municipality_week_keys",
        "maximum_key_repetitions"
    ],
    "value": [
        len(df),
        df["codigo_ibge"].nunique(),
        duplicados_postmerge,
        max_rep_postmerge
    ]
})

validacion_panel.to_csv(
    os.path.join(
        DATASETS_DIR,
        "dataset_multimodal_v5_panel_audit.csv"
    ),
    index=False
)

# ==================================================
# EXPORTAR
# ==================================================

print("\n===================================")
print("EXPORTANDO")
print("===================================")

df.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

df.to_csv(
    OUTPUT_CSV,
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