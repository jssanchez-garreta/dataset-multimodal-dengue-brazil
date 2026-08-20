# If you are in a Notebook (Jupyter, Google Colab or VS Code Notebooks)
!pip install fpdf2  

# ============================================================
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# BLOQUE 0:
# TABLA MAESTRA DE MUNICIPIOS DE BRASIL
#
# Fuente:
# BR_Municipios_2025.shp (IBGE)
# ============================================================

import geopandas as gpd
import pandas as pd

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

SHAPEFILE = "BR_Municipios_2025.shp"

OUTPUT_CSV = "municipios_brasil_master.csv"

# CRS geográfico oficial del IBGE
CRS_GEOGRAFICO = 4674  # SIRGAS 2000

# CRS proyectado para cálculo de centroides
CRS_PROYECTADO = 5880

# ------------------------------------------------------------
# CARGAR SHAPEFILE
# ------------------------------------------------------------

print("=" * 60)
print("CARGANDO SHAPEFILE")
print("=" * 60)

# IMPORTANTE:
# use_arrow=False evita un problema detectado en
# GeoPandas 1.1.3 + Pyogrio 0.12.1 donde miles de
# geometrías aparecen como None.

gdf = gpd.read_file(
    SHAPEFILE,
    use_arrow=False
)

print(f"Municipios cargados: {len(gdf):,}")

# ------------------------------------------------------------
# COMPROBAR CRS
# ------------------------------------------------------------

print("\nSistema de coordenadas:")

if gdf.crs is None:
    raise ValueError(
        "ERROR: No se ha detectado CRS. "
        "Comprueba que el archivo .prj existe."
    )

print(gdf.crs)

# ------------------------------------------------------------
# CALCULAR CENTROIDES
# ------------------------------------------------------------

print("\nCalculando centroides...")

gdf_proj = gdf.to_crs(CRS_PROYECTADO)

centroids = gdf_proj.centroid

centroids = gpd.GeoSeries(
    centroids,
    crs=CRS_PROYECTADO
).to_crs(CRS_GEOGRAFICO)

gdf["longitud"] = centroids.x
gdf["latitud"] = centroids.y

# ------------------------------------------------------------
# VALIDACIÓN DE GEOMETRÍAS
# ------------------------------------------------------------

if gdf["latitud"].isnull().sum() > 0:
    raise ValueError(
        "Existen centroides sin calcular. "
        "Revisar lectura del shapefile."
    )

if gdf["longitud"].isnull().sum() > 0:
    raise ValueError(
        "Existen centroides sin calcular. "
        "Revisar lectura del shapefile."
    )

# ------------------------------------------------------------
# CREAR TABLA MAESTRA
# ------------------------------------------------------------

master = gdf[
    [
        "CD_MUN",
        "NM_MUN",
        "CD_UF",
        "NM_UF",
        "SIGLA_UF",
        "AREA_KM2",
        "latitud",
        "longitud"
    ]
].copy()

master.columns = [
    "codigo_ibge",
    "municipio",
    "codigo_estado",
    "estado",
    "sigla_estado",
    "area_km2",
    "latitud",
    "longitud"
]

# ------------------------------------------------------------
# CÓDIGO IBGE COMO TEXTO
# ------------------------------------------------------------

master["codigo_ibge_str"] = (
    master["codigo_ibge"]
    .astype(str)
    .str.zfill(7)
)

# ------------------------------------------------------------
# MUNICIPIO + ESTADO
# ------------------------------------------------------------

master["municipio_estado"] = (
    master["municipio"]
    + " ("
    + master["sigla_estado"]
    + ")"
)

# ------------------------------------------------------------
# WKT DEL CENTROIDE
# ------------------------------------------------------------

master["centroide_wkt"] = (
    "POINT ("
    + master["longitud"].astype(str)
    + " "
    + master["latitud"].astype(str)
    + ")"
)

# ------------------------------------------------------------
# REORDENAR COLUMNAS
# ------------------------------------------------------------

master = master[
    [
        "codigo_ibge",
        "codigo_ibge_str",
        "municipio",
        "municipio_estado",
        "codigo_estado",
        "estado",
        "sigla_estado",
        "area_km2",
        "latitud",
        "longitud",
        "centroide_wkt"
    ]
]

# ------------------------------------------------------------
# VALIDACIONES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("VALIDACIONES")
print("=" * 60)

total_municipios = len(master)

municipios_unicos = master["codigo_ibge"].nunique()

duplicados = master["codigo_ibge"].duplicated().sum()

print(f"Municipios totales : {total_municipios:,}")
print(f"Municipios únicos  : {municipios_unicos:,}")
print(f"Duplicados         : {duplicados}")

print("\nValores nulos:")

print(master.isnull().sum())

# ------------------------------------------------------------
# RESUMEN
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)

print(
    f"Número de estados: "
    f"{master['sigla_estado'].nunique()}"
)

print(
    f"Área total (km²): "
    f"{master['area_km2'].sum():,.2f}"
)

print("\nMunicipios por estado:")

print(
    master["sigla_estado"]
    .value_counts()
    .sort_index()
)

# ------------------------------------------------------------
# GUARDAR CSV
# ------------------------------------------------------------

master.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# GUARDAR GEOPACKAGE
# ------------------------------------------------------------

gdf.to_file(
    "municipios_brasil_master.gpkg",
    driver="GPKG"
)

print("\nCSV generado correctamente:")
print(OUTPUT_CSV)

print("\nGeoPackage generado correctamente:")
print("municipios_brasil_master.gpkg")

# ------------------------------------------------------------
# PREVISUALIZACIÓN
# ------------------------------------------------------------

print("\nPrimeros registros:")

print(master.head())

print("\nProceso finalizado correctamente.")