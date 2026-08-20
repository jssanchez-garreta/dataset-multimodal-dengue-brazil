#!pip install -q pystac-client planetary-computer rasterstats
# ============================================================
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# VARIABLES AMBIENTALES MODIS
# 2010-2025
#
# PRODUCTOS UTILIZADOS:
#
#   MOD13Q1 v061
#       - NDVI
#       - EVI
#
#   MOD11A2 v061
#       - LST_Day
#       - LST_Night
#
# FUENTE:
#   Microsoft Planetary Computer
#
# COBERTURA ESPACIAL:
#   5573 municipios IBGE de Brasil
#
# COBERTURA TEMPORAL:
#   2010-01-01
#   2013-12-01
#
# ENTRADAS:
#
#   municipios_brasil_master.gpkg
#
#   municipio_tile_mapping.csv
#
# SALIDAS:
#
#   Brasil_YYYY-MM-DD.csv
#
# EJEMPLOS:
#
#   Brasil_2010-01-01.csv
#   Brasil_2010-02-01.csv
#   ...
#   Brasil_2025-12-01.csv
#
# COLUMNAS DE SALIDA:
#
#   CD_MUN
#   NM_MUN
#   SIGLA_UF
#   fecha
#
#   NDVI_mean
#   NDVI_std
#
#   EVI_mean
#   EVI_std
#
#   LST_Day_mean
#   LST_Day_std
#
#   LST_Night_mean
#   LST_Night_std
#
# CARACTERÍSTICAS:
#
#   - Procesamiento mensual automático
#   - Descarga automática desde Planetary Computer
#   - Reproyección automática al CRS MODIS
#   - Estadísticos zonales municipales
#   - Consolidación automática de municipios multi-tile
#   - Reintentos automáticos ante errores HTTP
#   - Continuación del procesamiento ante fallos de tesela
#   - Estimación de tiempos de ejecución
#
# ============================================================


from datetime import datetime, timedelta

import os
import time

import pandas as pd
import requests
import rasterio
import planetary_computer

from rasterstats import zonal_stats

from google.colab import drive

drive.mount('/content/drive')

# ==================================================
# DIRECTORIOS
# ==================================================

BASE_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL/MODIS_BRASIL"
)

LOG_DIR = os.path.join(
    BASE_DIR,
    "logs"
)

os.makedirs(
    BASE_DIR,
    exist_ok=True
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

# ==================================================
# PLANETARY COMPUTER
# ==================================================

import geopandas as gpd

from pystac_client import Client

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

# ==================================================
# MUNICIPIOS
# ==================================================

municipios = gpd.read_file(
    "municipios_brasil_master.gpkg"
)

print(
    f"Municipios: {len(municipios)}"
)

# ==================================================
# MAPPING MUNICIPIO-TILE
# ==================================================

mapping = pd.read_csv(
    "municipio_tile_mapping.csv"
)

print(
    f"Registros mapping: {len(mapping)}"
)

municipios["CD_MUN"] = (
    municipios["CD_MUN"]
    .astype(str)
)

mapping["CD_MUN"] = (
    mapping["CD_MUN"]
    .astype(str)
)

# ==================================================
# LISTA DE TESELAS
# ==================================================

tiles_brasil = sorted(
    mapping["tile"]
    .unique()
    .tolist()
)

print(
    f"Tiles Brasil: {len(tiles_brasil)}"
)

print(tiles_brasil)

# ==================================================
# CONFIGURACIÓN VARIABLES
# ==================================================

VARIABLES = {
    "NDVI": {
        "collection": "modis-13Q1-061",
        "asset": "250m_16_days_NDVI",
        "factor": 1 / 10000,
        "offset": 0,
        "nodata": -3000
    },

    "EVI": {
        "collection": "modis-13Q1-061",
        "asset": "250m_16_days_EVI",
        "factor": 1 / 10000,
        "offset": 0,
        "nodata": -3000
    },

    "LST_Day": {
        "collection": "modis-11A2-061",
        "asset": "LST_Day_1km",
        "factor": 0.02,
        "offset": -273.15,
        "nodata": 0
    },

    "LST_Night": {
        "collection": "modis-11A2-061",
        "asset": "LST_Night_1km",
        "factor": 0.02,
        "offset": -273.15,
        "nodata": 0
    }
}

# ==================================================
# LOGS
# ==================================================

LOG_FILE = os.path.join(
    LOG_DIR,
    "modis_2010_2010.log"
)

def write_log(msg):

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    with open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            f"[{timestamp}] {msg}\n"
        )

# ==================================================
# DESCARGA ROBUSTA
# ==================================================

def descargar_tiff(url, raster_file):

    for intento in range(3):

        try:

            r = requests.get(
                url,
                timeout=300
            )

            if r.status_code != 200:

                print(
                    f"      HTTP {r.status_code}"
                )

                time.sleep(5)

                continue

            content_type = r.headers.get(
                "content-type",
                ""
            )

            if (
                "xml" in content_type.lower()
                or
                "html" in content_type.lower()
            ):

                print(
                    "      Respuesta XML/HTML"
                )

                time.sleep(5)

                continue

            with open(
                raster_file,
                "wb"
            ) as f:

                f.write(r.content)

            if (
                os.path.getsize(
                    raster_file
                ) < 10000
            ):

                print(
                    "      TIFF demasiado pequeño"
                )

                time.sleep(5)

                continue

            rasterio.open(
                raster_file
            )

            return True

        except Exception as e:

            print(
                f"      Intento "
                f"{intento+1}/3"
            )

            print(e)

            time.sleep(5)

    return False

# ==================================================
# PROCESAR VARIABLE
# ==================================================

def procesar_variable(variable, fecha):

    cfg = VARIABLES[variable]

    fecha_ini = datetime.strptime(
        fecha,
        "%Y-%m-%d"
    )

    fecha_fin = fecha_ini + timedelta(days=1)

    rango = (
        f"{fecha_ini:%Y-%m-%d}/"
        f"{fecha_fin:%Y-%m-%d}"
    )

    print(f"\n{variable} - {fecha}")

    # ==========================================
    # CONSULTA ROBUSTA AL CATÁLOGO STAC
    # ==========================================

    items_fecha = None

    for intento in range(10):

        try:

            search = catalog.search(
                collections=[cfg["collection"]],
                datetime=rango
            )

            items_fecha = list(
                search.items()
            )

            if len(items_fecha) > 0:

                break

            print(
                f"Catálogo vacío "
                f"({intento+1}/10)"
            )

            time.sleep(60)

        except Exception as e:

            print(
                f"Error catálogo "
                f"({intento+1}/10)"
            )

            print(e)

            time.sleep(60)

    if not items_fecha:

        raise Exception(
            f"No se pudo acceder al "
            f"catálogo para {fecha}"
        )

    items_por_tile = {}

    for item in items_fecha:

        item = planetary_computer.sign(
            item
        )

        for tile in tiles_brasil:

            if tile in item.id:

                items_por_tile[
                    tile
                ] = item

    resultados = []

    for tile in tiles_brasil:

        try:

            if tile not in items_por_tile:
                continue

            item = items_por_tile[tile]

            print(
                f"  {variable}: {tile}"
            )

            url = item.assets[
                cfg["asset"]
            ].href

            raster_file = (
                f"{tile}_{variable}.tif"
            )

            ok = descargar_tiff(
                url,
                raster_file
            )

            if not ok:

                print(
                    f"      Tile omitido"
                )

                continue

            codigos = mapping.loc[
                mapping["tile"] == tile,
                "CD_MUN"
            ].unique()

            municipios_tile = municipios[
                municipios["CD_MUN"].isin(
                    codigos
                )
            ].copy()

            src = rasterio.open(
                raster_file
            )

            municipios_tile = (
                municipios_tile
                .to_crs(src.crs)
            )

            stats = zonal_stats(
                municipios_tile,
                raster_file,
                stats=[
                    "mean",
                    "std"
                ],
                nodata=cfg["nodata"]
            )

            out = pd.concat(
                [
                    municipios_tile[
                        [
                            "CD_MUN",
                            "NM_MUN",
                            "SIGLA_UF"
                        ]
                    ].reset_index(drop=True),

                    pd.DataFrame(stats)

                ],
                axis=1
            )

            src.close()

            if os.path.exists(raster_file):
                os.remove(raster_file)

            out[
                f"{variable}_mean"
            ] = (
                out["mean"]
                * cfg["factor"]
            ) + cfg["offset"]

            out[
                f"{variable}_std"
            ] = (
                out["std"]
                * cfg["factor"]
            )

            resultados.append(
                out[
                    [
                        "CD_MUN",
                        "NM_MUN",
                        "SIGLA_UF",
                        f"{variable}_mean",
                        f"{variable}_std"
                    ]
                ]
            )

        except Exception as e:

            print(
                f"      ERROR tile {tile}"
            )

            print(e)

            continue

    if len(resultados) == 0:

        raise Exception(
            f"No hay resultados "
            f"para {variable}"
        )

    final = pd.concat(
        resultados,
        ignore_index=True
    )

    final["fecha"] = fecha

    final = (
        final
        .groupby(
            [
                "CD_MUN",
                "NM_MUN",
                "SIGLA_UF",
                "fecha"
            ],
            as_index=False
        )
        .agg(
            **{
                f"{variable}_mean":
                (
                    f"{variable}_mean",
                    "mean"
                ),

                f"{variable}_std":
                (
                    f"{variable}_std",
                    "mean"
                )
            }
        )
    )

    print(
        f"{variable} completos: "
        f"{final[f'{variable}_mean'].notna().sum()}"
    )

    return final

# ==================================================
# PROCESAR MES
# ==================================================

def procesar_mes(fecha):

    year = fecha[:4]

    output_dir = os.path.join(
        BASE_DIR,
        year
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    archivo = os.path.join(
        output_dir,
        f"Brasil_{fecha}.csv"
    )

    if os.path.exists(archivo):

        print(
            f"\nYA EXISTE: {archivo}"
        )

        write_log(
            f"SALTADO {fecha}"
        )

        return False

    print("\n")
    print("=" * 60)
    print(f"PROCESANDO MES {fecha}")
    print("=" * 60)

    ndvi = procesar_variable(
        "NDVI",
        fecha
    )

    evi = procesar_variable(
        "EVI",
        fecha
    )

    lst_day = procesar_variable(
        "LST_Day",
        fecha
    )

    lst_night = procesar_variable(
        "LST_Night",
        fecha
    )

    final = (
        ndvi
        .merge(
            evi,
            on=[
                "CD_MUN",
                "NM_MUN",
                "SIGLA_UF",
                "fecha"
            ],
            how="outer"
        )
        .merge(
            lst_day,
            on=[
                "CD_MUN",
                "NM_MUN",
                "SIGLA_UF",
                "fecha"
            ],
            how="outer"
        )
        .merge(
            lst_night,
            on=[
                "CD_MUN",
                "NM_MUN",
                "SIGLA_UF",
                "fecha"
            ],
            how="outer"
        )
    )

    final.to_csv(
        archivo,
        index=False
    )

    write_log(
        f"GENERADO {archivo}"
    )

    print("\nARCHIVO GENERADO")
    print(archivo)

    print(
        f"Municipios: {len(final)}"
    )

    return True


# ==================================================
# FECHAS DE PRUEBA
# ==================================================

fechas = pd.date_range(
    start="2010-01-01",
    end="2010-12-01",
    freq="MS"
)

# ==================================================
# EJECUCIÓN
# ==================================================

inicio_global = time.time()

n_total = len(fechas)

for i, fecha in enumerate(fechas):

    fecha_str = fecha.strftime(
        "%Y-%m-%d"
    )

    inicio_mes = time.time()

    try:

        procesado = procesar_mes(
            fecha_str
        )

        if procesado:

            write_log(
                f"COMPLETADO {fecha_str}"
            )

    except Exception as e:

        print(
            f"\nERROR MES "
            f"{fecha_str}"
        )

        print(e)

        write_log(
            f"ERROR MES {fecha_str}: {e}"
        )

    tiempo_mes = (
        time.time()
        - inicio_mes
    ) / 60

    meses_hechos = i + 1

    tiempo_total = (
        time.time()
        - inicio_global
    ) / 60

    promedio = (
        tiempo_total
        / meses_hechos
    )

    restantes = (
        n_total
        - meses_hechos
    )

    estimado = (
        promedio
        * restantes
    )

    print("\n--------------------------------")

    print(
        f"Meses completados: "
        f"{meses_hechos}/{n_total}"
    )

    print(
        f"Tiempo mes: "
        f"{tiempo_mes:.2f} min"
    )

    print(
        f"Promedio: "
        f"{promedio:.2f} min/mes"
    )

    print(
        f"Tiempo restante estimado: "
        f"{estimado:.2f} min"
    )

    print("--------------------------------")

print("\nFINALIZADO")