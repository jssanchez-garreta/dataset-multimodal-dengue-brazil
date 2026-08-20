# ============================================================
# 10_build_dataset_enso.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Construcción del dataset ENSO
#
# VARIABLES:
#
#   nino34
#   soi
#
# FUENTES:
#
#   Niño 3.4
#   https://www.cpc.ncep.noaa.gov/data/indices/detrend.nino34.ascii.txt
#
#   SOI (Standardized)
#   https://www.cpc.ncep.noaa.gov/data/indices/soi
#
# SALIDAS:
#
#   enso_2010_2025.csv
#   enso_2010_2025.parquet
#
# ============================================================

import os
import requests

import pandas as pd

# ==================================================
# GOOGLE DRIVE
# ==================================================

from google.colab import drive

drive.mount("/content/drive")

# ==================================================
# DIRECTORIO
# ==================================================

BASE_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL/ENSO_BRASIL"
)

# ==================================================
# URLS NOAA
# ==================================================

NINO34_URL = (
    "https://www.cpc.ncep.noaa.gov/"
    "data/indices/detrend.nino34.ascii.txt"
)

SOI_URL = (
    "https://www.cpc.ncep.noaa.gov/"
    "data/indices/soi"
)

# ==================================================
# DESCARGA NINO34
# ==================================================

print("\n===================================")
print("DESCARGANDO NINO34")
print("===================================")

r = requests.get(
    NINO34_URL,
    timeout=60
)

r.raise_for_status()

with open(
    "nino34.txt",
    "wb"
) as f:

    f.write(r.content)

# ==================================================
# LECTURA NINO34
# ==================================================

nino = pd.read_csv(
    "nino34.txt",
    sep=r"\s+"
)

print(
    f"Registros NINO34: {len(nino)}"
)

print(
    "Columnas NINO34:"
)

print(
    nino.columns.tolist()
)

# ==================================================
# PREPARACIÓN NINO34
# ==================================================

nino = nino[
    [
        "YR",
        "MON",
        "ANOM"
    ]
].copy()

nino.columns = [
    "anio",
    "mes",
    "nino34"
]

# ==================================================
# DESCARGA SOI
# ==================================================

print("\n===================================")
print("DESCARGANDO SOI")
print("===================================")

r = requests.get(
    SOI_URL,
    timeout=60
)

r.raise_for_status()

texto_soi = r.text

# ==================================================
# EXTRAER BLOQUE STANDARDIZED DATA
# ==================================================

lineas = texto_soi.splitlines()

inicio = None

for i, linea in enumerate(lineas):

    if (
        "STANDARDIZED"
        in linea.upper()
    ):

        inicio = i
        break

if inicio is None:

    raise Exception(
        "No se encontró el bloque "
        "STANDARDIZED DATA."
    )

# ==================================================
# PARSEAR SOI STANDARDIZED
# ==================================================

datos = []

for linea in lineas[inicio:]:

    partes = linea.split()

    # buscamos solo filas tipo:
    #
    # 2010 -1.1 -1.5 ...
    #

    if len(partes) != 13:
        continue

    try:

        anio = int(
            partes[0]
        )

    except:

        continue

    # ignorar años futuros NOAA
    if anio > 2025:
        continue

    try:

        for mes in range(1, 13):

            valor = float(
                partes[mes]
            )

            # ignorar -999.9

            if valor <= -999:

                continue

            datos.append(
                [
                    anio,
                    mes,
                    valor
                ]
            )

    except:

        continue

soi = pd.DataFrame(
    datos,
    columns=[
        "anio",
        "mes",
        "soi"
    ]
)

print(
    f"Registros SOI: {len(soi)}"
)

# ==================================================
# VALIDAR DUPLICADOS SOI
# ==================================================

duplicados_soi = (
    soi
    .duplicated(
        subset=[
            "anio",
            "mes"
        ]
    )
    .sum()
)

print(
    f"Duplicados SOI: "
    f"{duplicados_soi}"
)

# ==================================================
# MERGE
# ==================================================

print("\n===================================")
print("UNIENDO SERIES")
print("===================================")

enso = nino.merge(
    soi,
    on=[
        "anio",
        "mes"
    ],
    how="inner"
)

# ==================================================
# FILTRAR PERIODO
# ==================================================

enso = enso[
    (
        enso["anio"] >= 2010
    )
    &
    (
        enso["anio"] <= 2025
    )
].copy()

# ==================================================
# FECHA
# ==================================================

enso["fecha"] = pd.to_datetime(
    dict(
        year=enso["anio"],
        month=enso["mes"],
        day=1
    )
)

enso = enso[
    [
        "fecha",
        "anio",
        "mes",
        "nino34",
        "soi"
    ]
]

enso = (
    enso
    .sort_values(
        "fecha"
    )
    .reset_index(
        drop=True
    )
)

# ==================================================
# VALIDACIÓN
# ==================================================

print("\n===================================")
print("VALIDACIÓN")
print("===================================")

print(
    f"Filas: {len(enso)}"
)

print(
    f"Primer mes: "
    f"{enso['fecha'].min().date()}"
)

print(
    f"Último mes: "
    f"{enso['fecha'].max().date()}"
)

print(
    f"Nulos NINO34: "
    f"{enso['nino34'].isna().sum()}"
)

print(
    f"Nulos SOI: "
    f"{enso['soi'].isna().sum()}"
)

duplicados = (
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
    f"{duplicados}"
)

# ==================================================
# EXPORTAR CSV
# ==================================================

CSV_OUT = os.path.join(
    BASE_DIR,
    "enso_2010_2025.csv"
)

enso.to_csv(
    CSV_OUT,
    index=False
)

# ==================================================
# EXPORTAR PARQUET
# ==================================================

PARQUET_OUT = os.path.join(
    BASE_DIR,
    "enso_2010_2025.parquet"
)

enso.to_parquet(
    PARQUET_OUT,
    index=False
)

# ==================================================
# RESUMEN FINAL
# ==================================================

print("\n===================================")
print("RESUMEN FINAL")
print("===================================")

print(
    f"Observaciones: {len(enso)}"
)

print(
    f"NINO34 min: "
    f"{enso['nino34'].min():.2f}"
)

print(
    f"NINO34 max: "
    f"{enso['nino34'].max():.2f}"
)

print(
    f"SOI min: "
    f"{enso['soi'].min():.2f}"
)

print(
    f"SOI max: "
    f"{enso['soi'].max():.2f}"
)

print("\nARCHIVOS GENERADOS")

print(CSV_OUT)

print(PARQUET_OUT)

print("\nFINALIZADO")