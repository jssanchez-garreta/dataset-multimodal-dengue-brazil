# ============================================================
# 18_build_dataset_demographics.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Construcción del dataset municipal demográfico
# a partir del Censo Demográfico 2022 (IBGE/SIDRA)
#
# FUENTE:
#
# SIDRA - Tabela 9514
# População residente por sexo
#
# SIDRA - Tabela 9515
# Índice de envelhecimento,
# Idade mediana,
# Razão de sexo
#
# RESULTADO:
#
# DEMOGRAPHICS_BRASIL/
#
# ├── demographics_brasil.csv
# └── demographics_brasil.parquet
#
# ============================================================

# ==================================================
# IMPORTS
# ==================================================

import os
import time

import numpy as np
import pandas as pd

from google.colab import drive

# ==================================================
# GOOGLE DRIVE
# ==================================================

if not os.path.ismount("/content/drive"):
    drive.mount("/content/drive")

# ==================================================
# CRONÓMETRO
# ==================================================

start_time = time.time()

# ==================================================
# DIRECTORIOS
# ==================================================

PROJECT_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL"
)

DEMOGRAPHICS_DIR = os.path.join(
    PROJECT_DIR,
    "DEMOGRAPHICS_BRASIL"
)

os.makedirs(
    DEMOGRAPHICS_DIR,
    exist_ok=True
)

# ==================================================
# SALIDAS
# ==================================================

OUTPUT_CSV = os.path.join(
    DEMOGRAPHICS_DIR,
    "demographics_brasil.csv"
)

OUTPUT_PARQUET = os.path.join(
    DEMOGRAPHICS_DIR,
    "demographics_brasil.parquet"
)

# ==================================================
# URLS SIDRA
# ==================================================

URL_POPULATION = (
    "https://apisidra.ibge.gov.br/values/"
    "t/9514/n6/all/v/allxp/p/all/"
    "c2/all/"
    "c287/100362/"
    "c286/113635"
)

URL_INDICATORS = (
    "https://apisidra.ibge.gov.br/values/"
    "t/9515/n6/all/v/all/p/all/"
    "d/v8845%202,v10612%202"
)

# ==================================================
# POBLACIÓN POR SEXO
# ==================================================

print("\n===================================")
print("DESCARGANDO TABLA 9514")
print("===================================")

pop = pd.read_json(
    URL_POPULATION
)

# eliminar fila de metadatos

pop = pop.iloc[1:].copy()

pop = pop.rename(
    columns={
        "D1C": "codigo_ibge",
        "D1N": "municipio",
        "D4N": "sexo",
        "V": "valor"
    }
)

pop["codigo_ibge"] = (
    pop["codigo_ibge"]
    .astype(str)
)

pop["valor"] = pd.to_numeric(
    pop["valor"],
    errors="coerce"
)

print(
    f"Registros población: "
    f"{len(pop):,}"
)

# ==================================================
# PIVOT SEXO
# ==================================================

sex_df = (
    pop[
        [
            "codigo_ibge",
            "sexo",
            "valor"
        ]
    ]
    .pivot_table(
        index="codigo_ibge",
        columns="sexo",
        values="valor",
        aggfunc="first"
    )
    .reset_index()
)

sex_df.columns.name = None

sex_df = sex_df.rename(
    columns={
        "Total":
            "population_total",
        "Homens":
            "male_population",
        "Mulheres":
            "female_population"
    }
)

sex_df[
    "male_population_pct"
] = (
    sex_df["male_population"]
    /
    sex_df["population_total"]
    * 100
)

sex_df[
    "female_population_pct"
] = (
    sex_df["female_population"]
    /
    sex_df["population_total"]
    * 100
)

# ==================================================
# TABLA 9515
# ==================================================

print("\n===================================")
print("DESCARGANDO TABLA 9515")
print("===================================")

ind = pd.read_json(
    URL_INDICATORS
)

ind = ind.iloc[1:].copy()

ind = ind.rename(
    columns={
        "D1C": "codigo_ibge",
        "D2N": "indicador",
        "V": "valor"
    }
)

ind["codigo_ibge"] = (
    ind["codigo_ibge"]
    .astype(str)
)

ind["valor"] = pd.to_numeric(
    ind["valor"],
    errors="coerce"
)

print(
    f"Registros indicadores: "
    f"{len(ind):,}"
)

# ==================================================
# PIVOT INDICADORES
# ==================================================

ind_df = (
    ind[
        [
            "codigo_ibge",
            "indicador",
            "valor"
        ]
    ]
    .pivot_table(
        index="codigo_ibge",
        columns="indicador",
        values="valor",
        aggfunc="first"
    )
    .reset_index()
)

ind_df.columns.name = None

ind_df = ind_df.rename(
    columns={
        "Índice de envelhecimento":
            "aging_index",

        "Idade mediana":
            "median_age",

        "Razão de sexo":
            "sex_ratio"
    }
)

# ==================================================
# MERGE
# ==================================================

print("\n===================================")
print("CONSOLIDANDO")
print("===================================")

demographics = sex_df.merge(
    ind_df,
    on="codigo_ibge",
    how="inner"
)

# ==================================================
# VARIABLES FINALES
# ==================================================

final_cols = [
    "codigo_ibge",

    "population_total",

    "male_population_pct",
    "female_population_pct",

    "aging_index",
    "median_age",
    "sex_ratio"
]

demographics = demographics[
    final_cols
].copy()

# ==================================================
# VALIDACIÓN
# ==================================================

print("\n===================================")
print("VALIDACIÓN")
print("===================================")

print(
    f"Municipios: "
    f"{len(demographics):,}"
)

duplicados = (
    demographics
    .duplicated(
        subset=["codigo_ibge"]
    )
    .sum()
)

print(
    f"Duplicados: "
    f"{duplicados}"
)

print("\nNULOS")

print(
    demographics
    .isna()
    .sum()
)

print("\nRESUMEN")

print(
    demographics[
        [
            "population_total",
            "male_population_pct",
            "female_population_pct",
            "aging_index",
            "median_age",
            "sex_ratio"
        ]
    ]
    .describe()
)

# ==================================================
# EXPORTAR
# ==================================================

print("\n===================================")
print("EXPORTANDO")
print("===================================")

demographics.to_csv(
    OUTPUT_CSV,
    index=False
)

demographics.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

# ==================================================
# TIEMPO
# ==================================================

elapsed = (
    time.time()
    - start_time
)

print(
    f"\nTiempo total: "
    f"{elapsed/60:.2f} minutos"
)

print("\nFINALIZADO")