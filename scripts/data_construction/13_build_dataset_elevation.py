# ============================================================
# 13_build_dataset_elevation.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Construcción del dataset municipal de elevación
#
# FUENTE:
#
# Copernicus DEM GLO-30
# (cop-dem-glo-30)
#
# SALIDAS:
#
# COPERNICUS_BRASIL/
# ├── elevation_partials/
# ├── elevation_processing_log.csv
# ├── elevation_brasil.csv
# └── elevation_brasil.parquet
#
# VARIABLES FINALES:
#
# CD_MUN
# elev_mean
# elev_min
# elev_max
# n_pixels
# n_tiles
#
# ============================================================

# ==================================================
# DEPENDENCIAS
# ==================================================

!pip install -q pystac-client planetary-computer rasterstats

# ==================================================
# IMPORTS
# ==================================================

import os
import time

from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd

from pystac_client import Client
import planetary_computer

from rasterstats import zonal_stats

# ==================================================
# GOOGLE DRIVE
# ==================================================

from google.colab import drive

drive.mount("/content/drive")

# ==================================================
# CRONÓMETRO
# ==================================================

start_time = time.time()

# ==================================================
# PROYECTO
# ==================================================

PROJECT_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL"
)

COPERNICUS_DIR = os.path.join(
    PROJECT_DIR,
    "COPERNICUS_BRASIL"
)

PARTIALS_DIR = os.path.join(
    COPERNICUS_DIR,
    "elevation_partials"
)

os.makedirs(
    PARTIALS_DIR,
    exist_ok=True
)

# ==================================================
# ARCHIVOS
# ==================================================

MUNICIPIOS_FILE = os.path.join(
    PROJECT_DIR,
    "municipios_brasil_master.gpkg"
)

MAPPING_FILE = os.path.join(
    COPERNICUS_DIR,
    "municipio_dem_tile_mapping.csv"
)

OUTPUT_CSV = os.path.join(
    COPERNICUS_DIR,
    "elevation_brasil.csv"
)

OUTPUT_PARQUET = os.path.join(
    COPERNICUS_DIR,
    "elevation_brasil.parquet"
)

LOG_FILE = os.path.join(
    COPERNICUS_DIR,
    "elevation_processing_log.csv"
)

# ==================================================
# MUNICIPIOS
# ==================================================

print("\n===================================")
print("LEYENDO MUNICIPIOS")
print("===================================")

municipios = gpd.read_file(
    MUNICIPIOS_FILE
)

municipios = municipios.to_crs(
    "EPSG:4326"
)

municipios["CD_MUN"] = (
    municipios["CD_MUN"]
    .astype(str)
)

print(
    f"Municipios: {len(municipios):,}"
)

# ==================================================
# MAPPING
# ==================================================

print("\n===================================")
print("LEYENDO MAPPING")
print("===================================")

mapping = pd.read_csv(
    MAPPING_FILE,
    dtype={
        "CD_MUN": str
    }
)

print(
    f"Relaciones municipio-tile: "
    f"{len(mapping):,}"
)

tiles = sorted(
    mapping["tile"].unique()
)

# ==========================================
# TEST
# ==========================================

#tiles = tiles[:5]

#print(
#    f"TEST MODE -> {len(tiles)} tiles"
#)

print(
    f"Tiles únicos: "
    f"{len(tiles):,}"
)

# ==================================================
# STAC
# ==================================================

print("\n===================================")
print("CONECTANDO STAC")
print("===================================")

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# ==================================================
# LOG EXISTENTE
# ==================================================

if os.path.exists(LOG_FILE):

    log_df = pd.read_csv(
        LOG_FILE
    )

    tiles_ok = set(
        log_df.loc[
            log_df["status"] == "OK",
            "tile"
        ]
    )

else:

    tiles_ok = set()

print(
    f"Tiles ya procesados: "
    f"{len(tiles_ok)}"
)

# ==================================================
# PROCESAMIENTO
# ==================================================

print("\n===================================")
print("PROCESAMIENTO")
print("===================================")

# --------------------------------------------------
# BUCLE PRINCIPAL
# --------------------------------------------------

for i, tile_id in enumerate(
    tiles,
    start=1
):

    print(
        f"\n[{i}/{len(tiles)}] {tile_id}"
    )

    partial_file = os.path.join(
        PARTIALS_DIR,
        f"{tile_id}.parquet"
    )

    if (
        tile_id in tiles_ok
        and
        os.path.exists(partial_file)
    ):

        print(
            "  Ya procesado"
        )

        continue

    try:

        # ==========================================
        # STAC ITEM
        # ==========================================

        search = catalog.search(
            collections=[
                "cop-dem-glo-30"
            ],
            ids=[
                tile_id
            ]
        )

        item = next(
            search.items()
        )

        item = planetary_computer.sign(
            item
        )

        raster_url = (
            item
            .assets["data"]
            .href
        )

        # ==========================================
        # MUNICIPIOS DEL TILE
        # ==========================================

        municipios_tile = mapping.loc[
            mapping["tile"] == tile_id,
            "CD_MUN"
        ].unique()

        subset = municipios[
            municipios["CD_MUN"]
            .isin(
                municipios_tile
            )
        ]

        print(
            f"  Municipios: "
            f"{len(subset)}"
        )

        # ==========================================
        # ZONAL STATS
        # ==========================================

        stats = zonal_stats(
            subset.geometry,
            raster_url,
            stats=[
                "mean",
                "min",
                "max",
                "count"
            ],
            nodata=-32767
        )

        # ==========================================
        # RESULTADOS DEL TILE
        # ==========================================

        resultados_tile = []

        for cd_mun, stat in zip(
            subset["CD_MUN"],
            stats
        ):

            if stat["count"] is None:
                continue

            if stat["count"] == 0:
                continue

            resultados_tile.append(
                {
                    "CD_MUN": cd_mun,
                    "tile": tile_id,
                    "mean_tile": stat["mean"],
                    "min_tile": stat["min"],
                    "max_tile": stat["max"],
                    "n_pixels": stat["count"]
                }
            )

        partial_df = pd.DataFrame(
            resultados_tile
        )

        partial_df.to_parquet(
            partial_file,
            index=False
        )

        print(
            f"  Guardado: "
            f"{len(partial_df)} registros"
        )

        # ==========================================
        # LOG
        # ==========================================

        log_row = pd.DataFrame(
            [
                {
                    "tile": tile_id,
                    "municipios": len(subset),
                    "status": "OK",
                    "timestamp":
                        datetime.now()
                }
            ]
        )

        if os.path.exists(LOG_FILE):

            log_row.to_csv(
                LOG_FILE,
                mode="a",
                header=False,
                index=False
            )

        else:

            log_row.to_csv(
                LOG_FILE,
                index=False
            )

    except Exception as e:

        print(
            f"  ERROR: {e}"
        )

        error_row = pd.DataFrame(
            [
                {
                    "tile": tile_id,
                    "municipios": 0,
                    "status":
                        f"ERROR: {e}",
                    "timestamp":
                        datetime.now()
                }
            ]
        )

        if os.path.exists(LOG_FILE):

            error_row.to_csv(
                LOG_FILE,
                mode="a",
                header=False,
                index=False
            )

        else:

            error_row.to_csv(
                LOG_FILE,
                index=False
            )

# ==================================================
# CONSOLIDACIÓN
# ==================================================

print("\n===================================")
print("CONSOLIDANDO")
print("===================================")

partial_files = sorted(
    [
        os.path.join(
            PARTIALS_DIR,
            f
        )
        for f in os.listdir(
            PARTIALS_DIR
        )
        if f.endswith(
            ".parquet"
        )
    ]
)

print(
    f"Parciales encontrados: "
    f"{len(partial_files):,}"
)

if len(partial_files) == 0:

    raise ValueError(
        "No se encontraron parciales."
    )

partials = pd.concat(
    [
        pd.read_parquet(f)
        for f in partial_files
    ],
    ignore_index=True
)

print(
    f"Registros parciales: "
    f"{len(partials):,}"
)

# ==================================================
# MEDIA PONDERADA
# ==================================================

partials["weighted_mean"] = (
    partials["mean_tile"]
    *
    partials["n_pixels"]
)

elev_mean = (
    partials
    .groupby("CD_MUN")
    .agg(
        weighted_sum=(
            "weighted_mean",
            "sum"
        ),
        pixel_sum=(
            "n_pixels",
            "sum"
        )
    )
)

elev_mean["elev_mean"] = (
    elev_mean["weighted_sum"]
    /
    elev_mean["pixel_sum"]
)

# ==================================================
# RESTO VARIABLES
# ==================================================

otros = (
    partials
    .groupby("CD_MUN")
    .agg(
        elev_min=(
            "min_tile",
            "min"
        ),

        elev_max=(
            "max_tile",
            "max"
        ),

        n_pixels=(
            "n_pixels",
            "sum"
        ),

        n_tiles=(
            "tile",
            "nunique"
        )
    )
)

# ==================================================
# DATASET FINAL
# ==================================================

elevation = (
    elev_mean[
        ["elev_mean"]
    ]
    .join(
        otros
    )
    .reset_index()
)

elevation = (
    elevation
    .sort_values(
        "CD_MUN"
    )
    .reset_index(
        drop=True
    )
)

elevation = elevation[
    [
        "CD_MUN",
        "elev_mean",
        "elev_min",
        "elev_max",
        "n_pixels",
        "n_tiles"
    ]
]

# ==================================================
# EXPORTAR
# ==================================================

print("\n===================================")
print("EXPORTANDO")
print("===================================")

elevation.to_csv(
    OUTPUT_CSV,
    index=False
)

elevation.to_parquet(
    OUTPUT_PARQUET,
    index=False
)

# ==================================================
# VALIDACIÓN
# ==================================================

print("\n===================================")
print("VALIDACIÓN")
print("===================================")

print(
    f"Municipios finales: "
    f"{len(elevation):,}"
)

print(
    f"Media elevación: "
    f"{elevation['elev_mean'].mean():.2f}"
)

print(
    f"Elevación mínima: "
    f"{elevation['elev_min'].min():.2f}"
)

print(
    f"Elevación máxima: "
    f"{elevation['elev_max'].max():.2f}"
)

print(
    f"Nulos elev_mean: "
    f"{elevation['elev_mean'].isna().sum()}"
)

print(
    f"Cobertura municipal: "
    f"{len(elevation)}/{len(municipios)}"
)

print(
    f"Municipios faltantes: "
    f"{len(municipios) - len(elevation)}"
)

print(
    f"Media tiles/municipio: "
    f"{elevation['n_tiles'].mean():.2f}"
)

print(
    f"Máximo tiles/municipio: "
    f"{elevation['n_tiles'].max()}"
)

# ==================================================
# ARCHIVOS
# ==================================================

print("\n===================================")
print("ARCHIVOS GENERADOS")
print("===================================")

print(OUTPUT_CSV)
print(OUTPUT_PARQUET)
print(LOG_FILE)

# ==================================================
# TIEMPO TOTAL
# ==================================================

elapsed = (
    time.time()
    -
    start_time
)

print(
    f"\nTiempo total: "
    f"{elapsed / 3600:.2f} horas"
)

print("\nFINALIZADO")