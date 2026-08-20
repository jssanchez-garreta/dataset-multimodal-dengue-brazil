# ============================================================
# 16_build_dataset_landcover.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Construcción del dataset municipal de uso y
# cobertura del suelo (Land Cover)
#
# FUENTE:
#
# ESA WorldCover 2021 v200
#
# RESULTADO:
#
# LANDCOVER_BRASIL/
#
# ├── landcover_partials/
# ├── landcover_processing_log.csv
# ├── landcover_brasil.csv
# └── landcover_brasil.parquet
#
# ============================================================

# ==================================================
# DEPENDENCIAS
# ==================================================

!pip install -q pystac-client planetary-computer rasterio rasterstats

# ==================================================
# IMPORTS
# ==================================================

import os
import time

from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd

import rasterio
from rasterio.mask import mask

from shapely.geometry import mapping

from pystac_client import Client
import planetary_computer

# ==================================================
# GOOGLE DRIVE
# ==================================================

from google.colab import drive
import os

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

LANDCOVER_DIR = os.path.join(
    PROJECT_DIR,
    "LANDCOVER_BRASIL"
)

PARTIALS_DIR = os.path.join(
    LANDCOVER_DIR,
    "landcover_partials"
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
    LANDCOVER_DIR,
    "municipio_worldcover_tile_mapping.csv"
)

OUTPUT_CSV = os.path.join(
    LANDCOVER_DIR,
    "landcover_brasil.csv"
)

OUTPUT_PARQUET = os.path.join(
    LANDCOVER_DIR,
    "landcover_brasil.parquet"
)

LOG_FILE = os.path.join(
    LANDCOVER_DIR,
    "landcover_processing_log.csv"
)

# ==================================================
# CLASES WORLDCOVER
# ==================================================

CLASS_MAP = {
    10: "tree_cover",
    20: "shrubland",
    30: "grassland",
    40: "cropland",
    50: "builtup",
    60: "bare_sparse",
    80: "water",
    90: "wetland",
    95: "mangroves"
}
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

mapping_df = pd.read_csv(
    MAPPING_FILE,
    dtype={
        "CD_MUN": str
    }
)

tiles = sorted(
    mapping_df["tile"].unique()
)

# ==========================================
# TEST MODE
# ==========================================

# tiles = tiles[:20]

# print(
#     f"TEST MODE -> {len(tiles)} tiles"
# )

print(
    f"Tiles utilizados: {len(tiles):,}"
)

# ==================================================
# STAC
# ==================================================

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# ==================================================
# LOG
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

# ==================================================
# PROCESAMIENTO
# ==================================================

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

        print("  Ya procesado")

        continue

    try:

        search = catalog.search(
            collections=[
                "esa-worldcover"
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
            .assets["map"]
            .href
        )

        municipios_tile = (
            mapping_df
            .loc[
                mapping_df["tile"] == tile_id,
                "CD_MUN"
            ]
            .unique()
        )

        subset = municipios[
            municipios["CD_MUN"]
            .isin(
                municipios_tile
            )
        ]

        print(
            f"  Municipios: {len(subset)}"
        )

        resultados = []

        errores = []

        with rasterio.open(
            raster_url
        ) as src:

            for _, row in subset.iterrows():

                try:

                    out, _ = mask(
                        src,
                        [
                            mapping(
                                row.geometry
                            )
                        ],
                        crop=True
                    )

                    arr = out[0]

                    arr = arr[
                        arr != 0
                    ]

                    if len(arr) == 0:

                        registro = {
                            "CD_MUN": row["CD_MUN"],
                            "tile": tile_id,
                            "n_pixels": 0
                        }

                        for code in CLASS_MAP:
                            registro[f"class_{code}"] = 0

                        resultados.append(registro)

                        continue

                    unique, counts = np.unique(
                        arr,
                        return_counts=True
                    )

                    total_pixels = (
                        counts.sum()
                    )

                    registro = {
                        "CD_MUN":
                            row["CD_MUN"],
                        "tile":
                            tile_id,
                        "n_pixels":
                            total_pixels
                    }

                    for code in CLASS_MAP:

                        registro[
                            f"class_{code}"
                        ] = (
                            counts[
                                unique == code
                            ].sum()
                            if code in unique
                            else 0
                        )

                    resultados.append(
                        registro
                    )

                except Exception as e:

                    errores.append(
                        {
                            "CD_MUN": row["CD_MUN"],
                            "tile": tile_id,
                            "error": str(e)
                        }
                    )

                    registro = {
                        "CD_MUN": row["CD_MUN"],
                        "tile": tile_id,
                        "n_pixels": 0
                    }

                    for code in CLASS_MAP:
                        registro[f"class_{code}"] = 0

                    resultados.append(
                        registro
                    )

                    continue

        partial_df = pd.DataFrame(
            resultados
        )

        errores_df = pd.DataFrame(
            errores
        )

        if len(errores_df) > 0:

            error_file = os.path.join(
                PARTIALS_DIR,
                f"{tile_id}_errors.csv"
            )

            errores_df.to_csv(
                error_file,
                index=False
            )

            print(
                f" ERRORES: "
                f"{len(errores_df)}"
            )

        partial_df.to_parquet(
            partial_file,
            index=False
        )

        print(
            f"  Guardado: "
            f"{len(partial_df)}"
        )

        log_row = pd.DataFrame(
            [
                {
                    "tile": tile_id,
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

land = pd.concat(
    [
        pd.read_parquet(f)
        for f in partial_files
    ],
    ignore_index=True
)

print(
    f"Registros parciales: "
    f"{len(land):,}"
)

# ==================================================
# AGREGACIÓN MUNICIPAL
# ==================================================

agg = (
    land
    .groupby("CD_MUN")
    .sum(
        numeric_only=True
    )
)

print(
    f"Municipios tras groupby: "
    f"{len(agg):,}"
)

agg["n_tiles"] = (
    land
    .groupby("CD_MUN")["tile"]
    .nunique()
)

print(
    f"Municipios tras calcular n_tiles: "
    f"{len(agg):,}"
)

# ==================================================
# PORCENTAJES
# ==================================================

for code, name in CLASS_MAP.items():

    agg[
        f"pct_{name}"
    ] = np.where(
        agg["n_pixels"] > 0,
        (
            agg[f"class_{code}"]
            /
            agg["n_pixels"]
            * 100
        ),
        0
    )

# ==================================================
# CLASE DOMINANTE
# ==================================================

pct_cols = [
    f"pct_{v}"
    for v in CLASS_MAP.values()
]

agg[
    "dominant_landcover"
] = (
    agg[pct_cols]
    .idxmax(axis=1)
    .str.replace(
        "pct_",
        "",
        regex=False
    )
)

# ==================================================
# DATASET FINAL
# ==================================================

final_cols = [
    "pct_tree_cover",
    "pct_shrubland",
    "pct_grassland",
    "pct_cropland",
    "pct_builtup",
    "pct_bare_sparse",
    "pct_water",
    "pct_wetland",
    "pct_mangroves",
    "dominant_landcover",
    "n_pixels",
    "n_tiles"
]
landcover = (
    agg[
        final_cols
    ]
    .reset_index()
)

print(
    f"Municipios dataset final: "
    f"{len(landcover):,}"
)

print(
    f"Municipios n_pixels = 0: "
    f"{(agg['n_pixels'] == 0).sum()}"
)

# ==================================================
# MUNICIPIOS FALTANTES
# ==================================================

municipios_total = set(
    municipios["CD_MUN"]
)

municipios_landcover = set(
    landcover["CD_MUN"]
)

faltantes = (
    municipios_total
    - municipios_landcover
)

print(
    f"Municipios faltantes: "
    f"{len(faltantes)}"
)

error_files = [
    f
    for f in os.listdir(
        PARTIALS_DIR
    )
    if f.endswith(
        "_errors.csv"
    )
]

print(
    f"Archivos de error: "
    f"{len(error_files)}"
)

if len(error_files) > 0:

    print("\nPrimeros archivos de error:")

    for f in sorted(error_files)[:10]:
        print(f)

if len(faltantes) > 0:

    print(
        sorted(
            list(faltantes)
        )[:50]
    )

# ==================================================
# VALIDACIÓN PORCENTAJES
# ==================================================

pct_cols = [
    "pct_tree_cover",
    "pct_shrubland",
    "pct_grassland",
    "pct_cropland",
    "pct_builtup",
    "pct_bare_sparse",
    "pct_water",
    "pct_wetland",
    "pct_mangroves"
]
landcover["pct_sum"] = (
    landcover[pct_cols]
    .sum(axis=1)
)

# ==================================================
# EXPORTAR
# ==================================================

landcover.to_csv(
    OUTPUT_CSV,
    index=False
)

landcover.to_parquet(
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
    f"Municipios: "
    f"{len(landcover):,}"
)

print(
    f"Nulos dominant_landcover: "
    f"{landcover['dominant_landcover'].isna().sum()}"
)

print(
    f"Media n_tiles: "
    f"{landcover['n_tiles'].mean():.2f}"
)

print("\nRESUMEN pct_sum")

print(
    landcover["pct_sum"]
    .describe()
)

print(
    f"\nMunicipios pct_sum < 99: "
    f"{(landcover['pct_sum'] < 99).sum()}"
)

print(
    "\nPeores municipios:"
)

print(
    landcover
    .loc[
        landcover["pct_sum"] < 99,
        [
            "CD_MUN",
            "pct_sum",
            "n_tiles"
        ]
    ]
    .sort_values(
        "pct_sum"
    )
    .head(20)
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
    f"{elapsed/3600:.2f} horas"
)

print("\nFINALIZADO")