# ============================================================
# 21_merge_population_density.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Incorporar population_density al dataset multimodal.
#
# ENTRADAS:
#
# dataset_multimodal_v5.parquet
# population_density_brasil.parquet
#
# SALIDAS:
#
# dataset_multimodal_v6.parquet
# dataset_multimodal_v6.csv
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

POPDENS_DIR = os.path.join(
    PROJECT_DIR,
    "POPULATION_DENSITY_BRASIL"
)

# ==================================================
# ARCHIVOS
# ==================================================

INPUT_DATASET = os.path.join(
    DATASETS_DIR,
    "dataset_multimodal_v5.parquet"
)

INPUT_POPDENS = os.path.join(
    POPDENS_DIR,
    "population_density_brasil.parquet"
)

OUTPUT_PARQUET = os.path.join(
    DATASETS_DIR,
    "dataset_multimodal_v6.parquet"
)

OUTPUT_CSV = os.path.join(
    DATASETS_DIR,
    "dataset_multimodal_v6.csv"
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
# LEER POPULATION DENSITY
# ==================================================

print("\n===================================")
print("LEYENDO POPULATION DENSITY")
print("===================================")

popdens = pd.read_parquet(
    INPUT_POPDENS
)

popdens["codigo_ibge"] = (
    popdens["codigo_ibge"]
    .astype(str)
)

print(
    f"Municipios: {len(popdens):,}"
)

# ==================================================
# VARIABLES A INCORPORAR
# ==================================================

popdens = popdens[
    [
        "codigo_ibge",
        "area_km2",
        "population_density"
    ]
]

# ==================================================
# VALIDACIÓN PRE-MERGE
# ==================================================

print("\n===================================")
print("VALIDACIÓN PRE-MERGE")
print("===================================")

duplicados = (
    popdens
    .duplicated(
        subset=["codigo_ibge"]
    )
    .sum()
)

print(
    f"Duplicados population density: "
    f"{duplicados}"
)

# ==================================================
# MERGE
# ==================================================

print("\n===================================")
print("MERGE")
print("===================================")

df = df.merge(
    popdens,
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
    f"Nulos area_km2: "
    f"{df['area_km2'].isna().sum():,}"
)

print(
    f"Nulos population_density: "
    f"{df['population_density'].isna().sum():,}"
)

# ==================================================
# RESUMEN
# ==================================================

print("\n===================================")
print("VALIDACIÓN VARIABLES")
print("===================================")

print(
    df[
        [
            "area_km2",
            "population_density"
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