# ============================================================
# 12_build_dem_tile_mapping.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Construir mapping:
#
# municipio <-> DEM tile
#
# FUENTE:
#
# Copernicus DEM GLO-30
#
# COLECCIÓN:
#
# cop-dem-glo-30
#
# SALIDA:
#
# municipio_dem_tile_mapping.csv
#
# ============================================================

import os

import pandas as pd
import geopandas as gpd

from shapely.geometry import shape

from pystac_client import Client

# ==================================================
# GOOGLE DRIVE
# ==================================================

from google.colab import drive

drive.mount("/content/drive")

# ==================================================
# DIRECTORIOS
# ==================================================

BASE_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL/COPERNICUS_BRASIL"
)

# ==================================================
# AJUSTAR ESTA RUTA SI FUERA NECESARIO
# ==================================================

MUNICIPIOS_FILE = (
    "/content/drive/MyDrive/DENGUE_BRASIL/"
    "municipios_brasil_master.gpkg"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "municipio_dem_tile_mapping.csv"
)

# ==================================================
# LEER MUNICIPIOS
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
    f"Municipios: {len(municipios)}"
)

# ==================================================
# CONEXIÓN STAC
# ==================================================

print("\n===================================")
print("CONECTANDO STAC")
print("===================================")

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# ==================================================
# BUSCAR TILES BRASIL
# ==================================================

print("\n===================================")
print("BUSCANDO TILES DEM")
print("===================================")

search = catalog.search(
    collections=[
        "cop-dem-glo-30"
    ],
    bbox=[
        -75,  # oeste
        -35,  # sur
        -29,  # este
        7     # norte
    ]
)

items = list(
    search.items()
)

print(
    f"Tiles encontrados: {len(items)}"
)

# ==================================================
# GDF DE TILES
# ==================================================

tiles = []

for item in items:

    tiles.append(
        {
            "tile": item.id,
            "geometry": shape(
                item.geometry
            )
        }
    )

tiles_gdf = gpd.GeoDataFrame(
    tiles,
    crs="EPSG:4326"
)

print(
    f"GeoDataFrame tiles: {len(tiles_gdf)}"
)

# ==================================================
# INTERSECCIÓN MUNICIPIOS-TILES
# ==================================================

print("\n===================================")
print("GENERANDO MAPPING")
print("===================================")

mapping = gpd.sjoin(
    municipios[
        [
            "CD_MUN",
            "geometry"
        ]
    ],
    tiles_gdf,
    how="inner",
    predicate="intersects"
)

mapping = mapping[
    [
        "CD_MUN",
        "tile"
    ]
].drop_duplicates()

# ==================================================
# ESTADÍSTICAS
# ==================================================

print(
    f"Registros mapping: {len(mapping):,}"
)

print(
    f"Municipios únicos: "
    f"{mapping['CD_MUN'].nunique():,}"
)

tiles_por_municipio = (
    mapping
    .groupby("CD_MUN")
    .size()
)

print(
    f"Tiles medio por municipio: "
    f"{tiles_por_municipio.mean():.2f}"
)

print(
    f"Tiles máximo municipio: "
    f"{tiles_por_municipio.max()}"
)

# ==================================================
# EXPORTAR
# ==================================================

mapping.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n===================================")
print("ARCHIVO GENERADO")
print("===================================")

print(
    OUTPUT_FILE
)

print("\nFINALIZADO")