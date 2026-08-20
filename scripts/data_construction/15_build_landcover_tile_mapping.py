# ============================================================
# 15_build_landcover_tile_mapping.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Asociar cada municipio brasileño únicamente con
# los tiles ESA WorldCover que realmente intersecta.
#
# FUENTE:
#
# ESA WorldCover 2021 v200
#
# RESULTADO:
#
# municipio_worldcover_tile_mapping.csv
#
# ============================================================

# ==================================================
# DEPENDENCIAS
# ==================================================

!pip install -q pystac-client planetary-computer

# ==================================================
# IMPORTS
# ==================================================

import os

import pandas as pd
import geopandas as gpd

from pystac_client import Client

# ==================================================
# GOOGLE DRIVE
# ==================================================

from google.colab import drive

drive.mount("/content/drive")

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

os.makedirs(
    LANDCOVER_DIR,
    exist_ok=True
)

# ==================================================
# ARCHIVOS
# ==================================================

MUNICIPIOS_FILE = os.path.join(
    PROJECT_DIR,
    "municipios_brasil_master.gpkg"
)

OUTPUT_FILE = os.path.join(
    LANDCOVER_DIR,
    "municipio_worldcover_tile_mapping.csv"
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
# STAC
# ==================================================

print("\n===================================")
print("BUSCANDO TILES WORLDCOVER")
print("===================================")

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

search = catalog.search(
    collections=["esa-worldcover"],
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

# ==================================================
# SOLO WORLDCOVER 2021 v200
# ==================================================

items = [
    item
    for item in items
    if "2021_v200" in item.id
]

print(
    f"Tiles WorldCover 2021 v200: "
    f"{len(items):,}"
)

# ==================================================
# GEOMETRÍAS DE TILES
# ==================================================

from shapely.geometry import shape

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
    geometry="geometry",
    crs="EPSG:4326"
)

# ==================================================
# VALIDACIÓN TILES
# ==================================================

print(
    tiles_gdf.head()
)

print(
    tiles_gdf.crs
)

print(
    f"GeoDataFrame tiles: "
    f"{len(tiles_gdf):,}"
)

# ==================================================
# INTERSECCIÓN ESPACIAL
# ==================================================

print("\n===================================")
print("CALCULANDO INTERSECCIONES")
print("===================================")

intersections = gpd.sjoin(
    municipios[
        ["CD_MUN", "geometry"]
    ],
    tiles_gdf,
    how="inner",
    predicate="intersects"
)

mapping = (
    intersections[
        [
            "CD_MUN",
            "tile"
        ]
    ]
    .drop_duplicates()
    .reset_index(
        drop=True
    )
)

# ==================================================
# EXPORTAR
# ==================================================

mapping.to_csv(
    OUTPUT_FILE,
    index=False
)

# ==================================================
# VALIDACIÓN
# ==================================================

print("\n===================================")
print("VALIDACIÓN")
print("===================================")

print(
    f"Relaciones municipio-tile: "
    f"{len(mapping):,}"
)

print(
    f"Municipios cubiertos: "
    f"{mapping['CD_MUN'].nunique():,}"
)

print(
    f"Tiles utilizados: "
    f"{mapping['tile'].nunique():,}"
)

tiles_por_municipio = (
    mapping
    .groupby("CD_MUN")
    .size()
)

print(
    f"Media tiles/municipio: "
    f"{tiles_por_municipio.mean():.2f}"
)

print(
    f"Máximo tiles/municipio: "
    f"{tiles_por_municipio.max()}"
)

# ==================================================
# ARCHIVOS
# ==================================================

print("\nARCHIVO GENERADO")

print(
    OUTPUT_FILE
)

# ==================================================
# MUNICIPIO FALTANTE
# ==================================================

municipios_mapping = set(
    mapping["CD_MUN"].unique()
)

municipios_total = set(
    municipios["CD_MUN"].unique()
)

faltantes = (
    municipios_total
    - municipios_mapping
)

print("\nMunicipios faltantes:")

for m in sorted(faltantes):
    print(m)

print("\nFINALIZADO")