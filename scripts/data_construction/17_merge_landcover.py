# ============================================================
# 17_merge_landcover.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Incorporar variables de Land Cover
# (ESA WorldCover 2021 v200)
# al dataset multimodal.
#
# ENTRADAS:
#
# dataset_multimodal_v3.parquet
# landcover_brasil.parquet
#
# SALIDAS:
#
# dataset_multimodal_v4.csv
# dataset_multimodal_v4.parquet
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

LANDCOVER_DIR = os.path.join(
    PROJECT_DIR,
    "LANDCOVER_BRASIL"
)

# ==================================================
# DATASET LANDCOVER
# ==================================================

LANDCOVER_FILE = os.path.join(
    LANDCOVER_DIR,
    "landcover_brasil.parquet"
)

# ==================================================
# DATASETS MASTER MULTIMODAL
# ==================================================

DATASETS_DIR = os.path.join(
    PROJECT_DIR,
    "DATASETS_MASTER_MULTIMODAL"
)

os.makedirs(
    DATASETS_DIR,
    exist_ok=True
)

INPUT_DATASET = os.path.join(
    DATASETS_DIR,
    "dataset_multimodal_v3.parquet"
)

OUTPUT_PARQUET = os.path.join(
    DATASETS_DIR,
    "dataset_multimodal_v4.parquet"
)

OUTPUT_CSV = os.path.join(
    DATASETS_DIR,
    "dataset_multimodal_v4.csv"
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
# CARGAR LAND COVER
# ==================================================

print("\n===================================")
print("LEYENDO LAND COVER")
print("===================================")

landcover = pd.read_parquet(
    LANDCOVER_FILE
)

landcover["CD_MUN"] = (
    landcover["CD_MUN"]
    .astype(str)
)

landcover = landcover.rename(
    columns={
        "CD_MUN": "codigo_ibge"
    }
)

print(
    f"Municipios: "
    f"{len(landcover):,}"
)

# ==================================================
# VARIABLES LAND COVER
# ==================================================

landcover_cols = [
    "codigo_ibge",
    "pct_tree_cover",
    "pct_shrubland",
    "pct_grassland",
    "pct_cropland",
    "pct_builtup",
    "pct_bare_sparse",
    "pct_water",
    "pct_wetland",
    "pct_mangroves",
    "dominant_landcover"
]

landcover = (
    landcover[
        landcover_cols
    ]
)

# ==================================================
# VALIDACIÓN PRE-MERGE
# ==================================================

print("\n===================================")
print("VALIDACIÓN PRE-MERGE")
print("===================================")

duplicados = (
    landcover
    .duplicated(
        subset=["codigo_ibge"]
    )
    .sum()
)

print(
    f"Duplicados landcover: "
    f"{duplicados}"
)

# ==================================================
# MERGE
# ==================================================

print("\n===================================")
print("MERGE")
print("===================================")

df = df.merge(
    landcover,
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

nulos_landcover = (
    df["pct_tree_cover"]
    .isna()
    .sum()
)

print(
    f"Nulos landcover: "
    f"{nulos_landcover:,}"
)

municipios_sin_landcover = (
    df.loc[
        df["pct_tree_cover"].isna(),
        "codigo_ibge"
    ]
    .nunique()
)

print(
    f"Municipios sin landcover: "
    f"{municipios_sin_landcover}"
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