# ============================================================
# 07_build_dataset_modis_master.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Construcción del dataset maestro MODIS
# a partir de todos los CSV mensuales generados
# entre 2010 y 2025.
#
# ENTRADA:
#
# MODIS_BRASIL/
#
# ├── 2010/
# ├── 2011/
# ├── ...
# ├── 2025/
#
# SALIDAS:
#
# MODIS_BRASIL_2010_2025_MASTER.csv
# MODIS_BRASIL_2010_2025_MASTER.parquet
#
# ============================================================

import os
import glob

import pandas as pd

# ==================================================
# GOOGLE DRIVE
# ==================================================

from google.colab import drive

drive.mount("/content/drive")

# ==================================================
# DIRECTORIO BASE
# ==================================================

BASE_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL/MODIS_BRASIL"
)

# ==================================================
# VALIDACIÓN DIRECTORIO
# ==================================================

print("\n===================================")
print("VALIDACIÓN DIRECTORIO")
print("===================================")

if not os.path.exists(BASE_DIR):

    raise Exception(
        f"No existe el directorio:\n{BASE_DIR}"
    )

print(f"Directorio encontrado:\n{BASE_DIR}")

# ==================================================
# BUSCAR TODOS LOS CSV
# ==================================================

archivos = sorted(
    glob.glob(
        os.path.join(
            BASE_DIR,
            "*",
            "Brasil_*.csv"
        )
    )
)

print("\n===================================")
print("ARCHIVOS ENCONTRADOS")
print("===================================")

for archivo in archivos:

    print(
        os.path.basename(archivo)
    )

print(
    f"\nTotal archivos encontrados: {len(archivos)}"
)

if len(archivos) == 0:

    raise Exception(
        "No se encontraron archivos CSV."
    )

# ==================================================
# RESUMEN POR AÑO
# ==================================================

archivos_por_anio = {}

for archivo in archivos:

    anio = os.path.basename(
        os.path.dirname(archivo)
    )

    archivos_por_anio[anio] = (
        archivos_por_anio.get(anio, 0) + 1
    )

print("\n===================================")
print("ARCHIVOS POR AÑO")
print("===================================")

for anio in sorted(archivos_por_anio):

    print(
        f"{anio}: "
        f"{archivos_por_anio[anio]}"
    )

# ==================================================
# LEER TODOS LOS CSV
# ==================================================

lista_df = []

print("\n===================================")
print("LEYENDO ARCHIVOS")
print("===================================")

for i, archivo in enumerate(
    archivos,
    start=1
):

    print(
        f"[{i}/{len(archivos)}] "
        f"{os.path.basename(archivo)}"
    )

    df = pd.read_csv(
        archivo,
        dtype={
            "CD_MUN": str
        }
    )

    lista_df.append(df)

# ==================================================
# CONCATENAR
# ==================================================

print("\n===================================")
print("CONCATENANDO")
print("===================================")

df_modis = pd.concat(
    lista_df,
    ignore_index=True
)

print(
    f"Filas concatenadas: {len(df_modis):,}"
)

# ==================================================
# FECHAS
# ==================================================

df_modis["fecha"] = pd.to_datetime(
    df_modis["fecha"]
)

df_modis["anio"] = (
    df_modis["fecha"]
    .dt.year
)

df_modis["mes"] = (
    df_modis["fecha"]
    .dt.month
)

# ==================================================
# ORDENAR
# ==================================================

df_modis = (
    df_modis
    .sort_values(
        [
            "CD_MUN",
            "fecha"
        ]
    )
    .reset_index(drop=True)
)

# ==================================================
# VALIDACIONES
# ==================================================

print("\n===================================")
print("VALIDACIÓN DATASET")
print("===================================")

print(
    f"Filas totales: {len(df_modis):,}"
)

print(
    f"Municipios únicos: "
    f"{df_modis['CD_MUN'].nunique():,}"
)

print(
    f"Meses únicos: "
    f"{df_modis['fecha'].nunique()}"
)

print(
    f"Primer mes: "
    f"{df_modis['fecha'].min().date()}"
)

print(
    f"Último mes: "
    f"{df_modis['fecha'].max().date()}"
)

# ==================================================
# DUPLICADOS
# ==================================================

duplicados = (
    df_modis
    .duplicated(
        subset=[
            "CD_MUN",
            "fecha"
        ]
    )
    .sum()
)

print(
    f"Duplicados CD_MUN+fecha: "
    f"{duplicados}"
)

# ==================================================
# ORDEN COLUMNAS
# ==================================================

columnas_base = [
    "CD_MUN",
    "NM_MUN",
    "SIGLA_UF",
    "fecha",
    "anio",
    "mes"
]

otras_columnas = [

    c for c in df_modis.columns

    if c not in columnas_base
]

df_modis = df_modis[
    columnas_base
    +
    otras_columnas
]

# ==================================================
# EXPORTAR CSV
# ==================================================

csv_out = os.path.join(
    BASE_DIR,
    "modis_brasil_2010_2025_master.csv"
)

print("\n===================================")
print("EXPORTANDO CSV")
print("===================================")

df_modis.to_csv(
    csv_out,
    index=False
)

print(
    f"CSV generado:\n{csv_out}"
)

# ==================================================
# EXPORTAR PARQUET
# ==================================================

parquet_out = os.path.join(
    BASE_DIR,
    "modis_brasil_2010_2025_master.parquet"
)

print("\n===================================")
print("EXPORTANDO PARQUET")
print("===================================")

df_modis.to_parquet(
    parquet_out,
    index=False
)

print(
    f"PARQUET generado:\n{parquet_out}"
)

# ==================================================
# RESUMEN FINAL
# ==================================================

print("\n===================================")
print("RESUMEN FINAL")
print("===================================")

print(
    f"Archivos procesados: {len(archivos)}"
)

print(
    f"Filas finales: {len(df_modis):,}"
)

print(
    f"Municipios únicos: "
    f"{df_modis['CD_MUN'].nunique()}"
)

print(
    f"Meses únicos: "
    f"{df_modis['fecha'].nunique()}"
)

print(
    f"Duplicados: "
    f"{duplicados}"
)

print("\nFINALIZADO")