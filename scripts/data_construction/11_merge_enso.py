# ============================================================
# 11_merge_enso.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Integrar:
#
# dataset_multimodal_v1.parquet
#
# +
#
# enso_2010_2025.parquet
#
# RESULTADO:
#
# dataset_multimodal_v2.csv
# dataset_multimodal_v2.parquet
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
# DIRECTORIO BASE
# ==================================================

BASE_DIR_MASTER = (
    "/content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL"
)

BASE_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL/ENSO_BRASIL"
)

# ==================================================
# ARCHIVOS
# ==================================================

MULTIMODAL_V1 = os.path.join(
    BASE_DIR_MASTER,
    "dataset_multimodal_v1.parquet"
)

ENSO_FILE = os.path.join(
    BASE_DIR,
    "enso_2010_2025.parquet"
)

CSV_OUT = os.path.join(
    BASE_DIR_MASTER,
    "dataset_multimodal_v2.csv"
)

PARQUET_OUT = os.path.join(
    BASE_DIR_MASTER,
    "dataset_multimodal_v2.parquet"
)

# ==================================================
# VALIDACIÓN
# ==================================================

print("\n===================================")
print("VALIDACIÓN ARCHIVOS")
print("===================================")

print(
    "dataset_multimodal_v1:",
    os.path.exists(MULTIMODAL_V1)
)

print(
    "enso_2010_2025:",
    os.path.exists(ENSO_FILE)
)

if not os.path.exists(MULTIMODAL_V1):

    raise Exception(
        f"No existe:\n{MULTIMODAL_V1}"
    )

if not os.path.exists(ENSO_FILE):

    raise Exception(
        f"No existe:\n{ENSO_FILE}"
    )

# ==================================================
# CARGA
# ==================================================

print("\n===================================")
print("CARGANDO DATASETS")
print("===================================")

df = pd.read_parquet(
    MULTIMODAL_V1
)

enso = pd.read_parquet(
    ENSO_FILE
)

print(
    f"dataset_multimodal_v1: {len(df):,}"
)

print(
    f"enso: {len(enso):,}"
)

# ==================================================
# VALIDACIÓN ENSO
# ==================================================

duplicados_enso = (
    enso
    .duplicated(
        subset=[
            "anio",
            "mes"
        ]
    )
    .sum()
)

print(
    f"Duplicados ENSO: "
    f"{duplicados_enso}"
)

# ==================================================
# COLUMNAS ENSO
# ==================================================

enso = enso[
    [
        "anio",
        "mes",
        "nino34",
        "soi"
    ]
]

# ==================================================
# MERGE
# ==================================================

print("\n===================================")
print("REALIZANDO MERGE")
print("===================================")

dataset_multimodal_v2 = df.merge(
    enso,
    on=[
        "anio",
        "mes"
    ],
    how="left"
)

print(
    f"Filas tras merge: "
    f"{len(dataset_multimodal_v2):,}"
)

# ==================================================
# COBERTURA ENSO
# ==================================================

print("\n===================================")
print("COBERTURA ENSO")
print("===================================")

for variable in [
    "nino34",
    "soi"
]:

    cobertura = (
        dataset_multimodal_v2[variable]
        .notna()
        .mean()
        * 100
    )

    print(
        f"{variable}: "
        f"{cobertura:.2f}%"
    )

# ==================================================
# ESTADÍSTICAS ENSO
# ==================================================

print("\n===================================")
print("ESTADÍSTICAS ENSO")
print("===================================")

print(
    f"NINO34 mínimo: "
    f"{dataset_multimodal_v2['nino34'].min():.2f}"
)

print(
    f"NINO34 máximo: "
    f"{dataset_multimodal_v2['nino34'].max():.2f}"
)

print(
    f"SOI mínimo: "
    f"{dataset_multimodal_v2['soi'].min():.2f}"
)

print(
    f"SOI máximo: "
    f"{dataset_multimodal_v2['soi'].max():.2f}"
)

# ==================================================
# EXPORTAR CSV
# ==================================================

print("\n===================================")
print("EXPORTANDO CSV")
print("===================================")

dataset_multimodal_v2.to_csv(
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

dataset_multimodal_v2.to_parquet(
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
    f"{len(dataset_multimodal_v2):,}"
)

print(
    f"Municipios: "
    f"{dataset_multimodal_v2['codigo_ibge'].nunique():,}"
)

print(
    f"Años: "
    f"{dataset_multimodal_v2['anio'].nunique()}"
)

print(
    f"Columnas: "
    f"{len(dataset_multimodal_v2.columns)}"
)

print("\nARCHIVOS GENERADOS")

print(CSV_OUT)

print(PARQUET_OUT)

print("\nFINALIZADO")