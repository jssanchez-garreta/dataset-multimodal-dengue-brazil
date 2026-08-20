# ============================================================
# 20_build_population_density.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Construir dataset municipal de densidad poblacional
#
# population_density =
# population_total / area_km2
#
# ENTRADAS:
#
# municipios_brasil_master.gpkg
# demographics_brasil.parquet
#
# SALIDAS:
#
# POPULATION_DENSITY_BRASIL/
#
# ├── population_density_brasil.csv
# └── population_density_brasil.parquet
#
# ============================================================

# ==================================================
# IMPORTS
# ==================================================

import os
import time

import geopandas as gpd
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

DEMOGRAPHICS_DIR = os.path.join(
    PROJECT_DIR,
    "DEMOGRAPHICS_BRASIL"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "POPULATION_DENSITY_BRASIL"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==================================================
# ARCHIVOS
# ==================================================

MUNICIPIOS_FILE = os.path.join(
    PROJECT_DIR,
    "municipios_brasil_master.gpkg"
)

DEMOGRAPHICS_FILE = os.path.join(
    DEMOGRAPHICS_DIR,
    "demographics_brasil.parquet"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "population_density_brasil.csv"
)

OUTPUT_PARQUET = os.path.join(
    OUTPUT_DIR,
    "population_density_brasil.parquet"
)

# ==================================================
# LEER MUNICIPIOS
# ==================================================

print("\n===================================")
print("LEYENDO MUNICIPIOS")
print("===================================")

gdf = gpd.read_file(
    MUNICIPIOS_FILE
)

print(
    f"Municipios: {len(gdf):,}"
)

# ==================================================
# EXTRAER ÁREA
# ==================================================

area_df = (
    gdf[
        [
            "CD_MUN",
            "NM_MUN",
            "SIGLA_UF",
            "AREA_KM2"
        ]
    ]
    .copy()
)

area_df = area_df.rename(
    columns={
        "CD_MUN": "codigo_ibge",
        "NM_MUN": "municipio",
        "SIGLA_UF": "uf",
        "AREA_KM2": "area_km2"
    }
)

area_df["codigo_ibge"] = (
    area_df["codigo_ibge"]
    .astype(str)
)

# ==================================================
# LEER DEMOGRAFÍA
# ==================================================

print("\n===================================")
print("LEYENDO DEMOGRAPHICS")
print("===================================")

demo = pd.read_parquet(
    DEMOGRAPHICS_FILE
)

demo["codigo_ibge"] = (
    demo["codigo_ibge"]
    .astype(str)
)

print(
    f"Municipios: {len(demo):,}"
)

# ==================================================
# VALIDACIÓN PRE-MERGE
# ==================================================

print("\n===================================")
print("VALIDACIÓN PRE-MERGE")
print("===================================")

duplicados_area = (
    area_df
    .duplicated(
        subset=["codigo_ibge"]
    )
    .sum()
)

duplicados_demo = (
    demo
    .duplicated(
        subset=["codigo_ibge"]
    )
    .sum()
)

print(
    f"Duplicados área: "
    f"{duplicados_area}"
)

print(
    f"Duplicados demographics: "
    f"{duplicados_demo}"
)

# ==================================================
# MERGE
# ==================================================

print("\n===================================")
print("MERGE")
print("===================================")

df = area_df.merge(
    demo[
        [
            "codigo_ibge",
            "population_total"
        ]
    ],
    on="codigo_ibge",
    how="inner"
)

# ==================================================
# DENSIDAD POBLACIONAL
# ==================================================

print("\n===================================")
print("CALCULANDO DENSIDAD")
print("===================================")

df["population_density"] = (
    df["population_total"]
    /
    df["area_km2"]
)

# ==================================================
# VALIDACIÓN
# ==================================================

print("\n===================================")
print("VALIDACIÓN")
print("===================================")

print(
    f"Municipios: {len(df):,}"
)

nulos = (
    df[
        [
            "area_km2",
            "population_total",
            "population_density"
        ]
    ]
    .isna()
    .sum()
)

print("\nNulos:")
print(nulos)

print("\nResumen:")

print(
    df[
        [
            "area_km2",
            "population_total",
            "population_density"
        ]
    ]
    .describe()
)

# ==================================================
# VARIABLES FINALES
# ==================================================

df = df[
    [
        "codigo_ibge",
        "municipio",
        "uf",
        "area_km2",
        "population_total",
        "population_density"
    ]
]

# ==================================================
# EXPORTAR
# ==================================================

print("\n===================================")
print("EXPORTANDO")
print("===================================")

df.to_csv(
    OUTPUT_CSV,
    index=False
)

df.to_parquet(
    OUTPUT_PARQUET,
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