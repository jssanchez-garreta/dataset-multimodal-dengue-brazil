# ============================================================
# 14_merge_elevation.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Integrar:
#
# dataset_multimodal_v2.parquet
#
# +
#
# elevation_brasil.parquet
#
# RESULTADO:
#
# dataset_multimodal_v3.csv
# dataset_multimodal_v3.parquet
#
# ============================================================

import os

import pandas as pd

# ==================================================
# GOOGLE DRIVE
# ==================================================

from google.colab import drive

drive.mount("/content/drive")

# ==================================================
# DIRECTORIOS
# ==================================================

BASE_DIR_MASTER = (
    "/content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL"
)

BASE_DIR_ELEVATION = (
    "/content/drive/MyDrive/DENGUE_BRASIL/COPERNICUS_BRASIL"
)

# ==================================================
# ARCHIVOS
# ==================================================

MULTIMODAL_V2 = os.path.join(
    BASE_DIR_MASTER,
    "dataset_multimodal_v2.parquet"
)

ELEVATION_FILE = os.path.join(
    BASE_DIR_ELEVATION,
    "elevation_brasil.parquet"
)

CSV_OUT = os.path.join(
    BASE_DIR_MASTER,
    "dataset_multimodal_v3.csv"
)

PARQUET_OUT = os.path.join(
    BASE_DIR_MASTER,
    "dataset_multimodal_v3.parquet"
)

# ==================================================
# VALIDACIÓN ARCHIVOS
# ==================================================

print("\n===================================")
print("VALIDACIÓN ARCHIVOS")
print("===================================")

print(
    "dataset_multimodal_v2:",
    os.path.exists(MULTIMODAL_V2)
)

print(
    "elevation_brasil:",
    os.path.exists(ELEVATION_FILE)
)

if not os.path.exists(MULTIMODAL_V2):

    raise Exception(
        f"No existe:\n{MULTIMODAL_V2}"
    )

if not os.path.exists(ELEVATION_FILE):

    raise Exception(
        f"No existe:\n{ELEVATION_FILE}"
    )

# ==================================================
# CARGA
# ==================================================

print("\n===================================")
print("CARGANDO DATASETS")
print("===================================")

df = pd.read_parquet(
    MULTIMODAL_V2
)

elevation = pd.read_parquet(
    ELEVATION_FILE
)

print(
    f"dataset_multimodal_v2: "
    f"{len(df):,}"
)

print(
    f"elevation: "
    f"{len(elevation):,}"
)

# ==================================================
# TIPOS
# ==================================================

df["codigo_ibge"] = (
    df["codigo_ibge"]
    .astype(str)
)

elevation["CD_MUN"] = (
    elevation["CD_MUN"]
    .astype(str)
)

# ==================================================
# VALIDACIÓN ELEVACIÓN
# ==================================================

print("\n===================================")
print("VALIDACIÓN ELEVACIÓN")
print("===================================")

duplicados = (
    elevation
    .duplicated(
        subset=["CD_MUN"]
    )
    .sum()
)

print(
    f"Duplicados CD_MUN: "
    f"{duplicados}"
)

print(
    f"Municipios únicos: "
    f"{elevation['CD_MUN'].nunique():,}"
)

# ==================================================
# COLUMNAS DE ELEVACIÓN
# ==================================================

elevation = elevation[
    [
        "CD_MUN",
        "elev_mean",
        "elev_min",
        "elev_max",
        "n_pixels",
        "n_tiles"
    ]
]

print(
    elevation.columns.tolist()
)

# ==================================================
# MERGE
# ==================================================

print("\n===================================")
print("REALIZANDO MERGE")
print("===================================")

dataset_multimodal_v3 = df.merge(
    elevation,
    left_on="codigo_ibge",
    right_on="CD_MUN",
    how="left"
)

dataset_multimodal_v3 = (
    dataset_multimodal_v3
    .drop(
        columns=["CD_MUN"]
    )
)

print(
    f"Filas tras merge: "
    f"{len(dataset_multimodal_v3):,}"
)

# ==================================================
# COBERTURA ELEVACIÓN
# ==================================================

print("\n===================================")
print("COBERTURA ELEVACIÓN")
print("===================================")

for variable in [
    "elev_mean",
    "elev_min",
    "elev_max",
    "n_pixels",
    "n_tiles"
]:

    cobertura = (
        dataset_multimodal_v3[variable]
        .notna()
        .mean()
        * 100
    )

    print(
        f"{variable}: "
        f"{cobertura:.2f}%"
    )

# ==================================================
# ESTADÍSTICAS
# ==================================================

print("\n===================================")
print("ESTADÍSTICAS ELEVACIÓN")
print("===================================")

print(
    f"Elevación media mínima: "
    f"{dataset_multimodal_v3['elev_mean'].min():.2f}"
)

print(
    f"Elevación media máxima: "
    f"{dataset_multimodal_v3['elev_mean'].max():.2f}"
)

print(
    f"Elevación mínima global: "
    f"{dataset_multimodal_v3['elev_min'].min():.2f}"
)

print(
    f"Elevación máxima global: "
    f"{dataset_multimodal_v3['elev_max'].max():.2f}"
)

print(
    f"Media tiles por municipio: "
    f"{dataset_multimodal_v3['n_tiles'].mean():.2f}"
)

# ==================================================
# MUNICIPIOS SIN ELEVACIÓN
# ==================================================

print("\n===================================")
print("VALIDACIÓN MUNICIPIOS")
print("===================================")

municipios_sin_elevacion = (
    dataset_multimodal_v3
    .loc[
        dataset_multimodal_v3["elev_mean"].isna(),
        "codigo_ibge"
    ]
    .nunique()
)

print(
    f"Municipios sin elevación: "
    f"{municipios_sin_elevacion}"
)

# ==================================================
# EXPORTAR CSV
# ==================================================

print("\n===================================")
print("EXPORTANDO CSV")
print("===================================")

dataset_multimodal_v3.to_csv(
    CSV_OUT,
    index=False
)

print(CSV_OUT)

# ==================================================
# EXPORTAR PARQUET
# ==================================================

print("\n===================================")
print("EXPORTANDO PARQUET")
print("===================================")

dataset_multimodal_v3.to_parquet(
    PARQUET_OUT,
    index=False
)

print(PARQUET_OUT)

# ==================================================
# RESUMEN FINAL
# ==================================================

print("\n===================================")
print("RESUMEN FINAL")
print("===================================")

print(
    f"Filas: "
    f"{len(dataset_multimodal_v3):,}"
)

print(
    f"Municipios: "
    f"{dataset_multimodal_v3['codigo_ibge'].nunique():,}"
)

print(
    f"Años: "
    f"{dataset_multimodal_v3['anio'].nunique()}"
)

print(
    f"Columnas: "
    f"{len(dataset_multimodal_v3.columns)}"
)

print("\nARCHIVOS GENERADOS")

print(CSV_OUT)

print(PARQUET_OUT)

print("\nFINALIZADO")