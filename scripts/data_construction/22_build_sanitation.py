# ============================================================
# 22_build_sanitation.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Construir dataset municipal de saneamiento
#
# FUENTE:
# https://www.aguaesaneamento.org.br
#
# Variables finales:
#
# water_supply_pct
# sewage_pct
# garbage_collection_pct
#
# ============================================================

# ==================================================
# IMPORTS
# ==================================================

import os
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import geopandas as gpd

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

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "SANITATION_BRASIL"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==================================================
# ARCHIVOS
# ==================================================

SANITATION_FILE = os.path.join(
    PROJECT_DIR,
    "archivo_exportado_sanitation.csv"
)

MUNICIPIOS_FILE = os.path.join(
    PROJECT_DIR,
    "municipios_brasil_master.gpkg"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "sanitation_brasil.csv"
)

OUTPUT_PARQUET = os.path.join(
    OUTPUT_DIR,
    "sanitation_brasil.parquet"
)

# ==================================================
# FUNCIONES
# ==================================================

def normalize_name(x):

    if pd.isna(x):
        return ""

    x = str(x)

    x = unicodedata.normalize(
        "NFKD",
        x
    )

    x = (
        x.encode("ASCII", "ignore")
         .decode("utf-8")
    )

    x = x.upper()

    x = re.sub(
        r"[^A-Z0-9 ]",
        " ",
        x
    )

    x = re.sub(
        r"\s+",
        " ",
        x
    )

    return x.strip()


def clean_pct(x):

    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    if x in ["--", "nan", "None"]:
        return np.nan

    if x == "Atendimento Pleno":
        return 0.0

    if x == "Não há":
        return np.nan

    x = (
        x.replace("%", "")
         .replace(",", ".")
         .strip()
    )

    try:
        return float(x)

    except:
        return np.nan


# ==================================================
# LEER CSV
# ==================================================

print("\n===================================")
print("LEYENDO SANEAMIENTO")
print("===================================")

san = pd.read_csv(
    SANITATION_FILE,
    sep=";"
)

print(
    f"Municipios: {len(san):,}"
)

# ==================================================
# LIMPIEZA TEXTO
# ==================================================

for col in san.columns:

    if san[col].dtype == object:

        san[col] = (
            san[col]
            .astype(str)
            .str.strip()
        )

# ==================================================
# VARIABLES ORIGINALES
# ==================================================

san["sem_agua_pct"] = (
    san["População sem Água"]
    .apply(clean_pct)
)

san["sem_esgoto_pct"] = (
    san["População sem Esgoto"]
    .apply(clean_pct)
)

san["sem_lixo_pct"] = (
    san["População sem coleta de lixo"]
    .apply(clean_pct)
)

# ==================================================
# VARIABLES FINALES
# ==================================================

san["water_supply_pct"] = (
    100
    - san["sem_agua_pct"]
)

san["sewage_pct"] = (
    100
    - san["sem_esgoto_pct"]
)

san["garbage_collection_pct"] = (
    100
    - san["sem_lixo_pct"]
)

# ==================================================
# LEER MUNICIPIOS MASTER
# ==================================================

print("\n===================================")
print("LEYENDO MUNICIPIOS MASTER")
print("===================================")

gdf = gpd.read_file(
    MUNICIPIOS_FILE
)

mun = (
    gdf[
        [
            "CD_MUN",
            "NM_MUN",
            "SIGLA_UF"
        ]
    ]
    .copy()
)

mun.rename(
    columns={
        "CD_MUN": "codigo_ibge",
        "NM_MUN": "Cidade",
        "SIGLA_UF": "UF"
    },
    inplace=True
)

mun["codigo_ibge"] = (
    mun["codigo_ibge"]
    .astype(str)
)

# ==================================================
# NORMALIZACIÓN
# ==================================================

san["merge_municipio"] = (
    san["Cidade"]
    .apply(normalize_name)
)

mun["merge_municipio"] = (
    mun["Cidade"]
    .apply(normalize_name)
)

san["UF"] = (
    san["UF"]
    .astype(str)
    .str.strip()
)

mun["UF"] = (
    mun["UF"]
    .astype(str)
    .str.strip()
)

# ==================================================
# MERGE
# ==================================================

print("\n===================================")
print("MATCH MUNICIPIOS")
print("===================================")

df = san.merge(
    mun[
        [
            "codigo_ibge",
            "Cidade",
            "UF",
            "merge_municipio"
        ]
    ],
    on=[
        "merge_municipio",
        "UF"
    ],
    how="left",
    suffixes=(
        "",
        "_ibge"
    )
)

# ==================================================
# CORRECCIONES MANUALES
# ==================================================

MANUAL_FIXES = {

    ("Açu", "RN"): "2400208",
    ("Amparo de São Francisco", "SE"): "2800100",
    ("Arês", "RN"): "2401206",
    ("Barão de Monte Alto", "MG"): "3105503",
    ("Dona Eusébia", "MG"): "3122904",
    ("Florínia", "SP"): "3516101",
    ("Muquém de São Francisco", "BA"): "2922250",
    ("Santo Antônio do Leverger", "MT"): "5107800",
    ("São Thomé das Letras", "MG"): "3165200",
    ("São Luís do Paraitinga", "SP"): "3550000",
    ("Augusto Severo", "RN"): "2401305",
    ("Fortaleza do Tabocão", "TO"): "1708254",
    ("São Luiz", "RR"): "1400605",
    ("Santa Teresinha", "BA"): "2928505",
}

for (municipio, uf), codigo in MANUAL_FIXES.items():

    mask = (
        (df["Cidade"] == municipio)
        & (df["UF"] == uf)
        & (df["codigo_ibge"].isna())
    )

    df.loc[mask, "codigo_ibge"] = codigo

# ==================================================
# VALIDACIÓN MATCH
# ==================================================

print(
    f"Municipios saneamiento: "
    f"{len(df):,}"
)

sin_codigo = (
    df["codigo_ibge"]
    .isna()
    .sum()
)

print(
    f"Sin código IBGE: "
    f"{sin_codigo}"
)

# ==================================================
# MOSTRAR NO MATCH
# ==================================================

if sin_codigo > 0:

    print("\nMunicipios sin match:")

    print(
        df.loc[
            df["codigo_ibge"].isna(),
            ["Cidade", "UF"]
        ]
        .sort_values(
            ["UF", "Cidade"]
        )
    )

# ==================================================
# VARIABLES FINALES
# ==================================================

df = df[
    [
        "codigo_ibge",
        "Cidade",
        "UF",
        "water_supply_pct",
        "sewage_pct",
        "garbage_collection_pct"
    ]
].copy()

df.rename(
    columns={
        "Cidade": "municipio",
        "UF": "uf"
    },
    inplace=True
)

# ==================================================
# VALIDACIÓN
# ==================================================

print("\n===================================")
print("VALIDACIÓN")
print("===================================")

print(
    f"Municipios finales: "
    f"{len(df):,}"
)

print("\nNULOS")

print(
    df[
        [
            "codigo_ibge",
            "water_supply_pct",
            "sewage_pct",
            "garbage_collection_pct"
        ]
    ]
    .isna()
    .sum()
)

print("\nRESUMEN")

print(
    df[
        [
            "water_supply_pct",
            "sewage_pct",
            "garbage_collection_pct"
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

df.to_csv(
    OUTPUT_CSV,
    index=False
)

df.to_parquet(
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