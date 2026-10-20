# ============================================================
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# BLOQUE 2
# CONSOLIDACIÓN CHIRPS 2010-2025
#
# Entradas:
#   CHIRPS_PROCESADO/
#       precipitacion_chirps_2010.parquet
#       ...
#       precipitacion_chirps_2025.parquet
#
#       validacion_chirps_2010.csv
#       ...
#       validacion_chirps_2025.csv
#
# Salidas:
#   CHIRPS_PROCESADO/
#       precipitacion_chirps_2010_2025.csv
#       precipitacion_chirps_2010_2025.parquet
#       validacion_chirps_global.csv
#
# ============================================================

import pandas as pd
import os

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

ANIOS = list(range(2010, 2026))

BASE_FOLDER = (
    "/content/drive/MyDrive/DENGUE_BRASIL/CHIRPS_PROCESADO"
)

OUTPUT_CSV = os.path.join(
    BASE_FOLDER,
    "precipitacion_chirps_2010_2025.csv"
)

OUTPUT_PARQUET = os.path.join(
    BASE_FOLDER,
    "precipitacion_chirps_2010_2025.parquet"
)

OUTPUT_VALIDACION = os.path.join(
    BASE_FOLDER,
    "validacion_chirps_global.csv"
)

OUTPUT_VALIDACION_CLAVES = os.path.join(
    BASE_FOLDER,
    "validacion_claves_municipio_semana.csv"
)

# ------------------------------------------------------------
# VALIDACIÓN ANUAL
# ------------------------------------------------------------

print("=" * 60)
print("VALIDACIÓN DE ARCHIVOS")
print("=" * 60)

validaciones = []

for anio in ANIOS:

    fichero = os.path.join(
        BASE_FOLDER,
        f"validacion_chirps_{anio}.csv"
    )

    if not os.path.exists(fichero):

        raise FileNotFoundError(
            f"No existe: {fichero}"
        )

    val = pd.read_csv(fichero)

    validaciones.append(val)

validacion_global = pd.concat(
    validaciones,
    ignore_index=True
)

print(validacion_global)

# ------------------------------------------------------------
# COMPROBACIONES
# ------------------------------------------------------------

if validacion_global["nulos"].sum() > 0:

    raise ValueError(
        "Existen años con valores nulos."
    )

if validacion_global["negativos"].sum() > 0:

    raise ValueError(
        "Existen años con precipitaciones negativas."
    )

print("\nValidación anual superada.")

# ------------------------------------------------------------
# CARGA DE PARQUETS
# ------------------------------------------------------------

print("\nCargando datasets anuales...")

datasets = []

for anio in ANIOS:

    fichero = os.path.join(
        BASE_FOLDER,
        f"precipitacion_chirps_{anio}.parquet"
    )

    if not os.path.exists(fichero):

        raise FileNotFoundError(
            f"No existe: {fichero}"
        )

    print(f"Cargando {anio}...")

    df = pd.read_parquet(fichero)

    # --------------------------------------------------------
    # ELIMINAR SEMANAS ISO QUE PERTENECEN A OTRO AÑO
    # --------------------------------------------------------

    df = df[
        df["SE"].str[:4].astype(int) == anio
    ].copy()

    datasets.append(df)

# ------------------------------------------------------------
# CONCATENACIÓN
# ------------------------------------------------------------

print("\nConcatenando datasets...")

dataset_final = pd.concat(
    datasets,
    ignore_index=True
)

# ------------------------------------------------------------
# ORDENAR
# ------------------------------------------------------------

dataset_final.sort_values(
    [
        "codigo_ibge",
        "SE"
    ],
    inplace=True
)
# ------------------------------------------------------------
# VALIDACIÓN GLOBAL
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("VALIDACIÓN GLOBAL")
print("=" * 60)

print(
    f"Filas: {len(dataset_final):,}"
)

print(
    f"Municipios: "
    f"{dataset_final['codigo_ibge'].nunique():,}"
)

print(
    f"Años: "
    f"{dataset_final['anio'].nunique()}"
)

print("\nNulos:")

print(
    dataset_final
    .isnull()
    .sum()
)

# ------------------------------------------------------------
# AUDITORÍA FINAL DEL BENCHMARK
# ------------------------------------------------------------

print("\nMunicipality-week uniqueness audit...")

duplicados = dataset_final.duplicated(
    subset=["codigo_ibge", "anio", "semana"]
).sum()

max_rep = (
    dataset_final
    .groupby(["codigo_ibge", "anio", "semana"])
    .size()
    .max()
)

print(f"Duplicated municipality-week keys: {duplicados}")
print(f"Maximum repetitions per key: {max_rep}")

# Resumen auditoría
validacion_clave = pd.DataFrame({
    "metric": [
        "total_rows",
        "unique_municipalities",
        "duplicated_municipality_week_keys",
        "maximum_key_repetitions"
    ],
    "value": [
        len(dataset_final),
        dataset_final["codigo_ibge"].nunique(),
        duplicados,
        max_rep
    ]
})

validacion_clave.to_csv(
    OUTPUT_VALIDACION_CLAVES,
    index=False
)

if duplicados > 0:
    raise ValueError(
        f"ERROR: Se detectaron {duplicados} claves municipio-semana duplicadas."
    )

print("OK: municipality-week key uniqueness validated.")

# ------------------------------------------------------------
# GUARDAR VALIDACIÓN
# ------------------------------------------------------------

validacion_global.to_csv(
    OUTPUT_VALIDACION,
    index=False
)

# ------------------------------------------------------------
# GUARDAR CSV
# ------------------------------------------------------------

print("\nGuardando CSV...")

dataset_final.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# GUARDAR PARQUET
# ------------------------------------------------------------

print("Guardando Parquet...")

dataset_final.to_parquet(
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
print(OUTPUT_VALIDACION)

print("\nProceso finalizado correctamente.")