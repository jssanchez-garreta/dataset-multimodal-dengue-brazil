# ============================================================
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# BLOQUE 1
# LIMPIEZA Y CONSTRUCCIÓN DEL DATASET EPIDEMIOLÓGICO
#
# Entrada:
#   epidemiologia_infodengue_2010_2025.csv
#
# Salidas:
#   epidemiologia_infodengue_2010_2025_clean.csv
#   epidemiologia_infodengue_2010_2025_clean.parquet
#
# ============================================================

import pandas as pd

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

INPUT_FILE = "epidemiologia_infodengue_2010_2025.csv"

OUTPUT_CSV = (
    "epidemiologia_infodengue_2010_2025_clean.csv"
)

OUTPUT_PARQUET = (
    "epidemiologia_infodengue_2010_2025_clean.parquet"
)

# ------------------------------------------------------------
# CARGAR DATASET
# ------------------------------------------------------------

print("=" * 60)
print("CARGANDO DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print(f"Filas: {len(df):,}")
print(f"Columnas: {len(df.columns)}")

# ------------------------------------------------------------
# ELIMINAR COLUMNAS INÚTILES
# ------------------------------------------------------------

columnas_inutiles = [
    "Localidade_id",
    "id",
    "versao_modelo",
    "tweet",
    "municipio_nome"
]

columnas_a_borrar = [
    col
    for col in columnas_inutiles
    if col in df.columns
]

df.drop(
    columns=columnas_a_borrar,
    inplace=True,
    errors="ignore"
)

print("\nColumnas eliminadas:")
print(columnas_a_borrar)

# ------------------------------------------------------------
# ELIMINAR COLUMNAS 100% VACÍAS
# ------------------------------------------------------------

columnas_vacias = []

for col in df.columns:

    if df[col].isnull().all():

        columnas_vacias.append(col)

if len(columnas_vacias) > 0:

    df.drop(
        columns=columnas_vacias,
        inplace=True
    )

print("\nColumnas vacías eliminadas:")
print(columnas_vacias)

# ------------------------------------------------------------
# VARIABLES TEMPORALES
# ------------------------------------------------------------

print("\nCreando variables temporales...")

df["SE"] = df["SE"].astype(str)

df["anio"] = (
    df["SE"]
    .str[:4]
    .astype(int)
)

df["semana"] = (
    df["SE"]
    .str[4:]
    .astype(int)
)

# ------------------------------------------------------------
# REORDENACIÓN
# ------------------------------------------------------------

columnas_temporales = [
    "codigo_ibge",
    "municipio",
    "estado",
    "data_iniSE",
    "SE",
    "anio",
    "semana"
]

columnas_restantes = [
    c
    for c in df.columns
    if c not in columnas_temporales
]

df = df[
    columnas_temporales +
    columnas_restantes
]

# ------------------------------------------------------------
# ORDENACIÓN
# ------------------------------------------------------------

df.sort_values(
    [
        "codigo_ibge",
        "SE"
    ],
    inplace=True
)

# ------------------------------------------------------------
# CALIDAD
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CALIDAD DEL DATASET")
print("=" * 60)

print(
    f"Filas finales: "
    f"{len(df):,}"
)

print(
    f"Columnas finales: "
    f"{len(df.columns)}"
)

print(
    f"Municipios únicos: "
    f"{df['codigo_ibge'].nunique():,}"
)

print("\nPorcentaje de valores nulos:")

nulos = (
    df.isnull()
      .mean()
      .mul(100)
      .round(2)
      .sort_values(
          ascending=False
      )
)

print(nulos)

# ------------------------------------------------------------
# GUARDAR CSV
# ------------------------------------------------------------

print("\nGuardando CSV...")

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# GUARDAR PARQUET
# ------------------------------------------------------------

print("Guardando Parquet...")

df.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

# ------------------------------------------------------------
# RESUMEN FINAL
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ARCHIVOS GENERADOS")
print("=" * 60)

print(OUTPUT_CSV)
print(OUTPUT_PARQUET)

print("\nColumnas finales:")

for col in df.columns:
    print(col)

print("\nProceso finalizado correctamente.")