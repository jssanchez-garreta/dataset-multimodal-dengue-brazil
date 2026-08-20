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