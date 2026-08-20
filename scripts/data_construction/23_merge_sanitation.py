# ============================================================
# 23_merge_sanitation.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Incorporar variables de saneamiento al dataset multimodal.
#
# ENTRADAS:
#
# dataset_multimodal_v6.parquet
# sanitation_brasil.parquet
#
# SALIDAS:
#
# dataset_multimodal_v7.parquet
# dataset_multimodal_v7.csv
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

SANITATION_DIR = os.path.join(
    PROJECT_DIR,
    "SANITATION_BRASIL"
)

# ==================================================
# ARCHIVOS
# ==================================================

INPUT_DATASET = os.path.join(
    DATASETS_DIR,
    "dataset_multimodal_v6.parquet"
)

INPUT_SANITATION = os.path.join(
    SANITATION_DIR,
    "sanitation_brasil.parquet"
)

OUTPUT_PARQUET = os.path.join(
    DATASETS_DIR,
    "dataset_multimodal_v7.parquet"
)

OUTPUT_CSV = os.path.join(
    DATASETS_DIR,
    "dataset_multimodal_v7.csv"
)

# ==================================================
# LEER DATASET MULTIMODAL
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
# LEER SANEAMIENTO
# ==================================================

print("\n===================================")
print("LEYENDO SANEAMIENTO")
print("===================================")

san = pd.read_parquet(
    INPUT_SANITATION
)

san["codigo_ibge"] = (
    san["codigo_ibge"]
    .astype(str)
)

print(
    f"Municipios: {len(san):,}"
)

# ==================================================
# VARIABLES A INCORPORAR
# ==================================================

san = san[
    [
        "codigo_ibge",
        "water_supply_pct",
        "sewage_pct",
        "garbage_collection_pct"
    ]
]

# ==================================================
# VALIDACIÓN PRE-MERGE
# ==================================================

print("\n===================================")
print("VALIDACIÓN PRE-MERGE")
print("===================================")

duplicados = (
    san
    .duplicated(
        subset=["codigo_ibge"]
    )
    .sum()
)

print(
    f"Duplicados saneamiento: "
    f"{duplicados}"
)

# ==================================================
# MERGE
# ==================================================

print("\n===================================")
print("MERGE")
print("===================================")

df = df.merge(
    san,
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

print(
    f"Nulos water_supply_pct: "
    f"{df['water_supply_pct'].isna().sum():,}"
)

print(
    f"Nulos sewage_pct: "
    f"{df['sewage_pct'].isna().sum():,}"
)

print(
    f"Nulos garbage_collection_pct: "
    f"{df['garbage_collection_pct'].isna().sum():,}"
)

# ==================================================
# MUNICIPIOS SIN SANEAMIENTO
# ==================================================

sin_saneamiento = (
    df.loc[
        df["water_supply_pct"].isna(),
        "codigo_ibge"
    ]
    .nunique()
)

print(
    f"Municipios sin dato de agua: "
    f"{sin_saneamiento}"
)

# ==================================================
# RESUMEN VARIABLES
# ==================================================

print("\n===================================")
print("VALIDACIÓN VARIABLES")
print("===================================")

print(
    df[
        [
            "water_supply_pct",
            "sewage_pct",
            "garbage_collection_pct"
        ]
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