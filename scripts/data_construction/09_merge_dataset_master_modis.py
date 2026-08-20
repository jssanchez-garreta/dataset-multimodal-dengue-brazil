# ============================================================
# 09_merge_dataset_master_modis.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Integrar:
#
# dataset_master_v1.parquet
#
# +
#
# modis_brasil_2010_2025_master.parquet
#
# RESULTADO:
#
# dataset_multimodal_v1.csv
# dataset_multimodal_v1.parquet
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
# DIRECTORIOS BASE
# ==================================================

BASE_DIR_MASTER = (
    "/content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL"
)

BASE_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL/MODIS_BRASIL"
)

# ==================================================
# ARCHIVOS
# ==================================================

MASTER_FILE = os.path.join(
    BASE_DIR_MASTER,
    "dataset_master_v1.parquet"
)

MODIS_FILE = os.path.join(
    BASE_DIR,
    "modis_brasil_2010_2025_master.parquet"
)

CSV_OUT = os.path.join(
    BASE_DIR_MASTER,
    "dataset_multimodal_v1.csv"
)

PARQUET_OUT = os.path.join(
    BASE_DIR_MASTER,
    "dataset_multimodal_v1.parquet"
)

# ==================================================
# VALIDACIÓN
# ==================================================

print("\n===================================")
print("VALIDACIÓN ARCHIVOS")
print("===================================")

print(
    "dataset_master_v1:",
    os.path.exists(MASTER_FILE)
)

print(
    "modis_master:",
    os.path.exists(MODIS_FILE)
)

if not os.path.exists(MASTER_FILE):

    raise Exception(
        f"No existe:\n{MASTER_FILE}"
    )

if not os.path.exists(MODIS_FILE):

    raise Exception(
        f"No existe:\n{MODIS_FILE}"
    )

# ==================================================
# CARGA
# ==================================================

print("\n===================================")
print("CARGANDO DATASETS")
print("===================================")

master = pd.read_parquet(
    MASTER_FILE
)

modis = pd.read_parquet(
    MODIS_FILE
)

print(
    f"Dataset master: {len(master):,} filas"
)

print(
    f"Dataset MODIS: {len(modis):,} filas"
)

# ==================================================
# PREPARACIÓN MASTER
# ==================================================

print("\n===================================")
print("PREPARANDO DATASET MASTER")
print("===================================")

master["codigo_ibge"] = (
    master["codigo_ibge"]
    .astype(str)
)

master["data_iniSE"] = pd.to_datetime(
    master["data_iniSE"]
)

master["anio"] = (
    master["data_iniSE"]
    .dt.year
)

master["mes"] = (
    master["data_iniSE"]
    .dt.month
)

print(
    f"Municipios master: "
    f"{master['codigo_ibge'].nunique():,}"
)

# ==================================================
# PREPARACIÓN MODIS
# ==================================================

print("\n===================================")
print("PREPARANDO MODIS")
print("===================================")

modis["CD_MUN"] = (
    modis["CD_MUN"]
    .astype(str)
)

modis = modis[
    [
        "CD_MUN",
        "anio",
        "mes",
        "NDVI_mean",
        "NDVI_std",
        "EVI_mean",
        "EVI_std",
        "LST_Day_mean",
        "LST_Day_std",
        "LST_Night_mean",
        "LST_Night_std"
    ]
]

duplicados_modis = (
    modis
    .duplicated(
        subset=[
            "CD_MUN",
            "anio",
            "mes"
        ]
    )
    .sum()
)

print(
    f"Duplicados MODIS: "
    f"{duplicados_modis}"
)

# ==================================================
# MERGE
# ==================================================

print("\n===================================")
print("REALIZANDO MERGE")
print("===================================")

dataset_multimodal = master.merge(
    modis,
    left_on=[
        "codigo_ibge",
        "anio",
        "mes"
    ],
    right_on=[
        "CD_MUN",
        "anio",
        "mes"
    ],
    how="left"
)

print(
    f"Filas tras merge: "
    f"{len(dataset_multimodal):,}"
)

# ==================================================
# LIMPIEZA
# ==================================================

if "CD_MUN" in dataset_multimodal.columns:

    dataset_multimodal.drop(
        columns=["CD_MUN"],
        inplace=True
    )

# ==================================================
# COBERTURA MODIS
# ==================================================

print("\n===================================")
print("COBERTURA MODIS")
print("===================================")

variables_modis = [
    "NDVI_mean",
    "EVI_mean",
    "LST_Day_mean",
    "LST_Night_mean"
]

for variable in variables_modis:

    cobertura = (
        dataset_multimodal[variable]
        .notna()
        .mean()
        * 100
    )

    print(
        f"{variable}: "
        f"{cobertura:.2f}%"
    )

# ==================================================
# MUNICIPIOS
# ==================================================

print("\n===================================")
print("COBERTURA MUNICIPIOS")
print("===================================")

mun_master = (
    dataset_multimodal["codigo_ibge"]
    .nunique()
)

mun_con_modis = (
    dataset_multimodal
    .loc[
        dataset_multimodal["NDVI_mean"]
        .notna(),
        "codigo_ibge"
    ]
    .nunique()
)

print(
    f"Municipios dataset master: "
    f"{mun_master:,}"
)

print(
    f"Municipios con MODIS: "
    f"{mun_con_modis:,}"
)

print(
    f"Municipios sin MODIS: "
    f"{mun_master - mun_con_modis:,}"
)

# ==================================================
# EXPORTACIÓN CSV
# ==================================================

print("\n===================================")
print("EXPORTANDO CSV")
print("===================================")

dataset_multimodal.to_csv(
    CSV_OUT,
    index=False
)

print(CSV_OUT)

# ==================================================
# EXPORTACIÓN PARQUET
# ==================================================

print("\n===================================")
print("EXPORTANDO PARQUET")
print("===================================")

dataset_multimodal.to_parquet(
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
    f"Filas finales: "
    f"{len(dataset_multimodal):,}"
)

print(
    f"Municipios únicos: "
    f"{dataset_multimodal['codigo_ibge'].nunique():,}"
)

print(
    f"Años: "
    f"{dataset_multimodal['anio'].nunique()}"
)

print(
    f"Columnas totales: "
    f"{len(dataset_multimodal.columns)}"
)

print("\nARCHIVOS GENERADOS")

print(CSV_OUT)

print(PARQUET_OUT)

print("\nFINALIZADO")