# ============================================================
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# BLOQUE 2
# PRECIPITACIÓN CHIRPS
#
# PROCESAMIENTO ANUAL
#
# ============================================================

import pandas as pd
import xarray as xr
import numpy as np
import os
import time

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

ANIO = 2024

MASTER_FILE = "municipios_brasil_master.csv"

CHIRPS_FOLDER = (
    "/content/drive/MyDrive/CHIRPS/"
)

CHIRPS_FILE = (
    CHIRPS_FOLDER
    + f"chirps-v2.0.{ANIO}.days_p05.nc"
)

OUTPUT_CSV = (
    f"precipitacion_chirps_{ANIO}.csv"
)

OUTPUT_PARQUET = (
    f"precipitacion_chirps_{ANIO}.parquet"
)

VALIDACION_FILE = (
    f"validacion_chirps_{ANIO}.csv"
)

# ------------------------------------------------------------
# COMPROBACIONES
# ------------------------------------------------------------

print("=" * 60)
print("CONFIGURACIÓN")
print("=" * 60)

print(CHIRPS_FILE)

if not os.path.exists(CHIRPS_FILE):

    raise FileNotFoundError(
        f"No existe:\n{CHIRPS_FILE}"
    )

print("Archivo CHIRPS encontrado.")

# ------------------------------------------------------------
# MUNICIPIOS
# ------------------------------------------------------------

print("\nCargando municipios...")

master = pd.read_csv(
    MASTER_FILE
)

master = master[
    [
        "codigo_ibge",
        "municipio_estado",
        "latitud",
        "longitud"
    ]
].copy()

print(
    f"Municipios: {len(master):,}"
)

# ------------------------------------------------------------
# ABRIR CHIRPS
# ------------------------------------------------------------

print("\nAbriendo CHIRPS...")

inicio_total = time.time()

ds = xr.open_dataset(
    CHIRPS_FILE
)

print("CHIRPS abierto.")

# ------------------------------------------------------------
# EXTRACCIÓN
# ------------------------------------------------------------

resultados = []

total = len(master)

for idx, row in master.iterrows():

    codigo = row["codigo_ibge"]

    municipio = row["municipio_estado"]

    lat = row["latitud"]

    lon = row["longitud"]

    if (idx + 1) % 250 == 0:

        print(
            f"{idx+1}/{total}"
        )

    try:

        serie = ds["precip"].sel(
            latitude=lat,
            longitude=lon,
            method="nearest"
        )

        df = (
            serie
            .to_dataframe()
            .reset_index()
        )

        df["codigo_ibge"] = codigo

        resultados.append(df)

    except Exception:

        continue

# ------------------------------------------------------------
# UNIÓN
# ------------------------------------------------------------

print("\nUnificando...")

precip = pd.concat(
    resultados,
    ignore_index=True
)

# ------------------------------------------------------------
# FECHAS
# ------------------------------------------------------------

precip["fecha"] = pd.to_datetime(
    precip["time"]
)

iso = (
    precip["fecha"]
    .dt.isocalendar()
)

precip["anio"] = iso.year.astype(int)
precip["semana"] = iso.week.astype(int)

precip["SE"] = (
    precip["anio"].astype(str)
    + precip["semana"]
      .astype(str)
      .str.zfill(2)
)

# ------------------------------------------------------------
# TRATAMIENTO DE NULOS CHIRPS
# ------------------------------------------------------------

precip["precip"] = (
    precip["precip"]
    .fillna(0)
)

precip["dia_lluvia"] = (
    precip["precip"] > 1
).astype(int)

# ------------------------------------------------------------
# AGREGACIÓN SEMANAL
# ------------------------------------------------------------

print("\nAgregando semanalmente...")

precip_semanal = (
    precip
    .groupby(
        [
            "codigo_ibge",
            "anio",
            "semana",
            "SE"
        ],
        as_index=False
    )
    .agg(
        precip_total_semana=(
            "precip",
            "sum"
        ),
        precip_media_semana=(
            "precip",
            "mean"
        ),
        precip_max_semana=(
            "precip",
            "max"
        ),
        dias_lluvia_semana=(
            "dia_lluvia",
            "sum"
        )
    )
)

# ------------------------------------------------------------
# VALIDACIÓN
# ------------------------------------------------------------

print("\n" + "=" * 60)
print(f"VALIDACIÓN CHIRPS {ANIO}")
print("=" * 60)

municipios = (
    precip_semanal["codigo_ibge"]
    .nunique()
)

semanas = (
    precip_semanal["semana"]
    .nunique()
)

filas = len(precip_semanal)

nulos = (
    precip_semanal
    .isnull()
    .sum()
    .sum()
)

negativos = (
    precip_semanal[
        "precip_total_semana"
    ] < 0
).sum()

print(f"Municipios: {municipios:,}")
print(f"Semanas: {semanas}")
print(f"Filas: {filas:,}")
print(f"Nulos: {nulos}")
print(f"Negativos: {negativos}")

if nulos > 0:
    raise ValueError(
        f"Existen {nulos} valores nulos."
    )

if negativos > 0:
    raise ValueError(
        f"Existen {negativos} precipitaciones negativas."
    )

# ------------------------------------------------------------
# GUARDAR VALIDACIÓN
# ------------------------------------------------------------

pd.DataFrame({
    "anio": [ANIO],
    "municipios": [municipios],
    "semanas": [semanas],
    "filas": [filas],
    "nulos": [nulos],
    "negativos": [negativos]
}).to_csv(
    VALIDACION_FILE,
    index=False
)

# ------------------------------------------------------------
# GUARDAR CSV
# ------------------------------------------------------------

print("\nGuardando CSV...")

precip_semanal.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# GUARDAR PARQUET
# ------------------------------------------------------------

print("Guardando Parquet...")

precip_semanal.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

# ------------------------------------------------------------
# RESUMEN FINAL
# ------------------------------------------------------------

fin_total = time.time()

print("\n" + "=" * 60)
print("ARCHIVOS GENERADOS")
print("=" * 60)

print(OUTPUT_CSV)
print(OUTPUT_PARQUET)
print(VALIDACION_FILE)

print(
    f"\nTiempo total: "
    f"{round((fin_total - inicio_total)/60,2)} min"
)

print("\nProceso finalizado.")