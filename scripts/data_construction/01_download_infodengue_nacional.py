# ============================================================
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# BLOQUE 1
# DESCARGA NACIONAL INFODENGUE
#
# Entrada:
#   municipios_brasil_master.csv
#
# Salidas:
#   epidemiologia_infodengue_2010_2025.csv
#   errores_descarga_infodengue.csv
#   backup_infodengue.csv
#
# ============================================================

import pandas as pd
import time

# ------------------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------------------

MASTER_FILE = "municipios_brasil_master.csv"

OUTPUT_FILE = "epidemiologia_infodengue_2010_2025.csv"

ERROR_FILE = "errores_descarga_infodengue.csv"

BACKUP_FILE = "backup_infodengue.csv"

YEAR_START = 2010
YEAR_END = 2025

GUARDADO_CADA = 100

PAUSA_SEGUNDOS = 0.1

# ------------------------------------------------------------
# CARGAR MUNICIPIOS
# ------------------------------------------------------------

print("=" * 60)
print("CARGANDO MUNICIPIOS")
print("=" * 60)

master = pd.read_csv(
    MASTER_FILE,
    dtype={"codigo_ibge_str": str}
)

print(f"Municipios: {len(master):,}")

# ------------------------------------------------------------
# CONTENEDORES
# ------------------------------------------------------------

datasets = []

errores = []

# ------------------------------------------------------------
# DESCARGA
# ------------------------------------------------------------

for idx, row in master.iterrows():

    codigo = row["codigo_ibge"]
    municipio = row["municipio"]
    estado = row["sigla_estado"]

    print(
        f"[{idx+1}/{len(master)}] "
        f"{municipio} ({estado})"
    )

    url = (
        "https://info.dengue.mat.br/api/alertcity?"
        f"geocode={codigo}"
        "&disease=dengue"
        "&format=csv"
        "&ew_start=1"
        "&ew_end=53"
        f"&ey_start={YEAR_START}"
        f"&ey_end={YEAR_END}"
    )

    try:

        df = pd.read_csv(url)

        if len(df) > 0:

            df["codigo_ibge"] = codigo
            df["municipio"] = municipio
            df["estado"] = estado

            datasets.append(df)

        else:

            errores.append({
                "codigo_ibge": codigo,
                "municipio": municipio,
                "estado": estado,
                "error": "sin_datos"
            })

    except Exception as e:

        errores.append({
            "codigo_ibge": codigo,
            "municipio": municipio,
            "estado": estado,
            "error": str(e)
        })

    # --------------------------------------------------------
    # BACKUP PERIÓDICO
    # --------------------------------------------------------

    if (idx + 1) % GUARDADO_CADA == 0:

        print(
            f"\nBACKUP: "
            f"{idx+1} municipios procesados"
        )

        if len(datasets) > 0:

            parcial = pd.concat(
                datasets,
                ignore_index=True
            )

            parcial.to_csv(
                BACKUP_FILE,
                index=False,
                encoding="utf-8-sig"
            )

            print(
                f"Backup actualizado: "
                f"{BACKUP_FILE}"
            )

    # --------------------------------------------------------
    # PAUSA
    # --------------------------------------------------------

    time.sleep(PAUSA_SEGUNDOS)

# ------------------------------------------------------------
# CONCATENACIÓN FINAL
# ------------------------------------------------------------

print("\nConcatenando datasets...")

dataset_final = pd.concat(
    datasets,
    ignore_index=True
)

# ------------------------------------------------------------
# GUARDAR DATASET
# ------------------------------------------------------------

dataset_final.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# GUARDAR ERRORES
# ------------------------------------------------------------

pd.DataFrame(errores).to_csv(
    ERROR_FILE,
    index=False,
    encoding="utf-8-sig"
)

# ------------------------------------------------------------
# RESUMEN
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)

print(
    f"Filas descargadas: "
    f"{len(dataset_final):,}"
)

print(
    f"Municipios descargados: "
    f"{dataset_final['codigo_ibge'].nunique():,}"
)

print(
    f"Errores: "
    f"{len(errores)}"
)

print("\nArchivos generados:")

print(OUTPUT_FILE)
print(ERROR_FILE)

print("\nProceso finalizado.")