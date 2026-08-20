# ============================================================
# 08_build_dataset_modis_quality_report.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Generar informe de calidad del dataset MODIS maestro.
#
# ENTRADA:
#
# MODIS_BRASIL_2010_2025_MASTER.parquet
#
# SALIDAS:
#
# modis_brasil_2010_2025_master.csv
# modis_brasil_2010_2025_master.parquet
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

BASE_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL/MODIS_BRASIL"
)

# ==================================================
# ARCHIVO DE ENTRADA
# ==================================================

master_file = os.path.join(
    BASE_DIR,
    "modis_brasil_2010_2025_master.parquet"
)

print("\n===================================")
print("CARGANDO DATASET MODIS MASTER")
print("===================================")

df = pd.read_parquet(master_file)

print(
    f"Filas: {len(df):,}"
)

# ==================================================
# INFORME MENSUAL
# ==================================================

print("\n===================================")
print("GENERANDO INFORME")
print("===================================")

report = (
    df.groupby(
        [
            "fecha",
            "anio",
            "mes"
        ],
        as_index=False
    )
    .agg(
        municipios=(
            "CD_MUN",
            "nunique"
        ),

        NDVI_validos=(
            "NDVI_mean",
            lambda x: x.notna().sum()
        ),

        EVI_validos=(
            "EVI_mean",
            lambda x: x.notna().sum()
        ),

        LST_Day_validos=(
            "LST_Day_mean",
            lambda x: x.notna().sum()
        ),

        LST_Night_validos=(
            "LST_Night_mean",
            lambda x: x.notna().sum()
        )
    )
)

# ==================================================
# PORCENTAJES
# ==================================================

report["NDVI_pct"] = (
    report["NDVI_validos"]
    / report["municipios"]
    * 100
)

report["EVI_pct"] = (
    report["EVI_validos"]
    / report["municipios"]
    * 100
)

report["LST_Day_pct"] = (
    report["LST_Day_validos"]
    / report["municipios"]
    * 100
)

report["LST_Night_pct"] = (
    report["LST_Night_validos"]
    / report["municipios"]
    * 100
)

# ==================================================
# ORDENAR
# ==================================================

report = report.sort_values(
    "fecha"
).reset_index(
    drop=True
)

# ==================================================
# EXPORTAR CSV
# ==================================================

csv_out = os.path.join(
    BASE_DIR,
    "modis_brasil_quality_report.csv"
)

report.to_csv(
    csv_out,
    index=False
)

# ==================================================
# EXPORTAR PARQUET
# ==================================================

parquet_out = os.path.join(
    BASE_DIR,
    "modis_brasil_quality_report.parquet"
)

report.to_parquet(
    parquet_out,
    index=False
)

# ==================================================
# RESUMEN GENERAL
# ==================================================

print("\n===================================")
print("RESUMEN GENERAL")
print("===================================")

print(
    f"Meses analizados: {len(report)}"
)

print(
    f"\nCobertura media NDVI: "
    f"{report['NDVI_pct'].mean():.2f}%"
)

print(
    f"Cobertura media EVI: "
    f"{report['EVI_pct'].mean():.2f}%"
)

print(
    f"Cobertura media LST_Day: "
    f"{report['LST_Day_pct'].mean():.2f}%"
)

print(
    f"Cobertura media LST_Night: "
    f"{report['LST_Night_pct'].mean():.2f}%"
)

# ==================================================
# PEORES MESES
# ==================================================

print("\n===================================")
print("PEOR MES POR VARIABLE")
print("===================================")

ndvi_min = report.loc[
    report["NDVI_pct"].idxmin()
]

evi_min = report.loc[
    report["EVI_pct"].idxmin()
]

lst_day_min = report.loc[
    report["LST_Day_pct"].idxmin()
]

lst_night_min = report.loc[
    report["LST_Night_pct"].idxmin()
]

print(
    f"NDVI      -> "
    f"{ndvi_min['fecha'].date()} "
    f"({ndvi_min['NDVI_pct']:.2f}%)"
)

print(
    f"EVI       -> "
    f"{evi_min['fecha'].date()} "
    f"({evi_min['EVI_pct']:.2f}%)"
)

print(
    f"LST_Day   -> "
    f"{lst_day_min['fecha'].date()} "
    f"({lst_day_min['LST_Day_pct']:.2f}%)"
)

print(
    f"LST_Night -> "
    f"{lst_night_min['fecha'].date()} "
    f"({lst_night_min['LST_Night_pct']:.2f}%)"
)

# ==================================================
# ARCHIVOS GENERADOS
# ==================================================

print("\n===================================")
print("ARCHIVOS GENERADOS")
print("===================================")

print(csv_out)
print(parquet_out)

print("\nFINALIZADO")