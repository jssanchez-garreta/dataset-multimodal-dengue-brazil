# ============================================================
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# DATASET_03_MASTER_V1
#
# UNE:
#   Dataset_01_Epidemiologia
#   Dataset_02_Precipitacion_CHIRPS
#
# ============================================================

import pandas as pd
import os
import time

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

INPUT_EPIDEMIOLOGIA = (
    "/content/drive/MyDrive/DENGUE_BRASIL/"
    "epidemiologia_infodengue_2010_2025_clean.parquet"
)

INPUT_PRECIPITACION = (
    "/content/drive/MyDrive/DENGUE_BRASIL/CHIRPS_PROCESADO/"
    "precipitacion_chirps_2010_2025.parquet"
)

OUTPUT_PARQUET = (
    "/content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL/"
    "dataset_master_v1.parquet"
)

OUTPUT_CSV = (
    "/content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL/"
    "dataset_master_v1.csv"
)

# ------------------------------------------------------------
# INICIO
# ------------------------------------------------------------

inicio = time.time()

print("=" * 60)
print("DATASET MASTER V1")
print("=" * 60)

# ------------------------------------------------------------
# COMPROBACIÓN DE ARCHIVOS
# ------------------------------------------------------------

print("\nComprobando archivos...")

print("\nEpidemiología:")

print(INPUT_EPIDEMIOLOGIA)

if not os.path.exists(
    INPUT_EPIDEMIOLOGIA
):
    raise FileNotFoundError(
        INPUT_EPIDEMIOLOGIA
    )

print(
    "Existe:",
    os.path.exists(
        INPUT_EPIDEMIOLOGIA
    )
)

print(
    "Tamaño:",
    f"{os.path.getsize(INPUT_EPIDEMIOLOGIA):,}"
)

print("\nPrecipitación:")

print(INPUT_PRECIPITACION)

if not os.path.exists(
    INPUT_PRECIPITACION
):
    raise FileNotFoundError(
        INPUT_PRECIPITACION
    )

print(
    "Existe:",
    os.path.exists(
        INPUT_PRECIPITACION
    )
)

print(
    "Tamaño:",
    f"{os.path.getsize(INPUT_PRECIPITACION):,}"
)

# ------------------------------------------------------------
# CARGAR EPIDEMIOLOGÍA
# ------------------------------------------------------------

print("\nCargando epidemiología...")

epi = pd.read_parquet(
    INPUT_EPIDEMIOLOGIA
)

print(
    f"Filas epidemiología: "
    f"{len(epi):,}"
)

print(
    f"Columnas epidemiología: "
    f"{len(epi.columns)}"
)

# ------------------------------------------------------------
# CARGAR PRECIPITACIÓN
# ------------------------------------------------------------

print("\nCargando precipitación...")

prec = pd.read_parquet(
    INPUT_PRECIPITACION
)

print(
    f"Filas precipitación: "
    f"{len(prec):,}"
)

print(
    f"Columnas precipitación: "
    f"{len(prec.columns)}"
)

# ------------------------------------------------------------
# NORMALIZAR CLAVES
# ------------------------------------------------------------

print("\nNormalizando claves...")

epi["codigo_ibge"] = (
    epi["codigo_ibge"]
    .astype(int)
)

prec["codigo_ibge"] = (
    prec["codigo_ibge"]
    .astype(int)
)

epi["SE"] = (
    epi["SE"]
    .astype(str)
)

prec["SE"] = (
    prec["SE"]
    .astype(str)
)


# ------------------------------------------------------------
# ELIMINAR SEMANAS SIN CONTRAPARTIDA EN CHIRPS
# ------------------------------------------------------------

SE_EXCLUIDAS = [
    "201453",
    "202553"
]

filas_antes = len(epi)

epi = epi[
    ~epi["SE"].isin(SE_EXCLUIDAS)
].copy()

filas_despues = len(epi)

print(
    f"\nFilas eliminadas por semanas sin CHIRPS: "
    f"{filas_antes - filas_despues:,}"
)

print(
    f"Filas epidemiología tras limpieza: "
    f"{filas_despues:,}"
)

# ------------------------------------------------------------
# COLUMNAS DE PRECIPITACIÓN
# ------------------------------------------------------------

columnas_prec = [
    "codigo_ibge",
    "SE",
    "precip_total_semana",
    "precip_media_semana",
    "precip_max_semana",
    "dias_lluvia_semana"
]

prec = prec[columnas_prec]

# ------------------------------------------------------------
# MERGE
# ------------------------------------------------------------

print("\nUniendo datasets...")

master = epi.merge(
    prec,
    on=[
        "codigo_ibge",
        "SE"
    ],
    how="left"
)

# ------------------------------------------------------------
# VALIDACIÓN
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("VALIDACIÓN MASTER V1")
print("=" * 60)

print(
    f"Filas: "
    f"{len(master):,}"
)

print(
    f"Columnas: "
    f"{len(master.columns)}"
)

print(
    f"Municipios: "
    f"{master['codigo_ibge'].nunique():,}"
)

# ------------------------------------------------------------
# NULOS
# ------------------------------------------------------------

print("\nPorcentaje de nulos:")

nulos = (
    master
    .isnull()
    .mean()
    .mul(100)
    .round(3)
    .sort_values(
        ascending=False
    )
)

print(nulos)

# ------------------------------------------------------------
# VALIDAR MERGE
# ------------------------------------------------------------

nulos_prec = (
    master[
        [
            "precip_total_semana",
            "precip_media_semana",
            "precip_max_semana",
            "dias_lluvia_semana"
        ]
    ]
    .isnull()
    .sum()
)

print(
    "\nNulos en variables de precipitación:"
)

print(nulos_prec)

# ------------------------------------------------------------
# ORDENAR
# ------------------------------------------------------------

master.sort_values(
    [
        "codigo_ibge",
        "SE"
    ],
    inplace=True
)

# ------------------------------------------------------------
# GUARDAR PARQUET
# ------------------------------------------------------------

print("\nGuardando Parquet...")

master.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

# ------------------------------------------------------------
# GUARDAR CSV
# ------------------------------------------------------------

print("Guardando CSV...")

master.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# RESUMEN FINAL
# ------------------------------------------------------------

fin = time.time()

print("\n" + "=" * 60)
print("ARCHIVOS GENERADOS")
print("=" * 60)

print(OUTPUT_PARQUET)
print(OUTPUT_CSV)

print(
    f"\nTiempo total: "
    f"{round((fin - inicio)/60, 2)} min"
)

print("\nProceso finalizado correctamente.")