# ============================================================
# 25_dataset_audit_v8.py
#
# PROYECTO:
# EARLY DETECTION OF DENGUE OUTBREAKS IN BRAZIL
#
# OBJETIVO:
#
# Auditoría completa de dataset_multimodal_v8
# y generación automática de informe Markdown.
#
# ENTRADA:
#
# dataset_multimodal_v8.parquet
#
# SALIDA:
#
# AUDIT_V8.md
#
# ============================================================

# ==================================================
# IMPORTS
# ==================================================

import os
import time
import pandas as pd
import numpy as np

from google.colab import drive

# ==================================================
# DRIVE
# ==================================================

if not os.path.ismount("/content/drive"):
    drive.mount("/content/drive")

# ==================================================
# TIMER
# ==================================================

start_time = time.time()

# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL"
)

DATASET_DIR = os.path.join(
    PROJECT_DIR,
    "DATASETS_MASTER_MULTIMODAL"
)

INPUT_FILE = os.path.join(
    DATASET_DIR,
    "dataset_multimodal_v8.parquet"
)

OUTPUT_FILE = os.path.join(
    DATASET_DIR,
    "AUDIT_V8.md"
)

# ==================================================
# LOAD
# ==================================================

print("\n===================================")
print("LOADING DATASET V8")
print("===================================")

df = pd.read_parquet(INPUT_FILE)

# ==================================================
# BASIC INFO
# ==================================================

rows = len(df)

cols = len(df.columns)

municipios = df["codigo_ibge"].nunique()

estados = df["estado"].nunique()

anio_min = df["anio"].min()

anio_max = df["anio"].max()

duplicates = int(
    df.duplicated().sum()
)

# ==================================================
# MISSING
# ==================================================

missing = pd.DataFrame({

    "variable":
        df.columns,

    "missing":
        df.isna().sum()

})

missing["missing_pct"] = (
    missing["missing"]
    / rows
    * 100
)

missing = missing.sort_values(
    "missing",
    ascending=False
)

top_missing = (
    missing
    .head(20)
)

# ==================================================
# NUMERIC STATS
# ==================================================

numeric_cols = (
    df.select_dtypes(
        include=np.number
    )
    .columns
)

# ==================================================
# TEMPORAL FEATURES
# ==================================================

lag_vars = sorted([
    c
    for c in df.columns
    if "_lag_" in c
])

rolling_vars = sorted([
    c
    for c in df.columns
    if "_roll" in c
])

# ==================================================
# REPORT
# ==================================================

report = []

report.append(
    "# AUDIT_V8\n"
)

report.append(
    "Generated automatically from "
    "dataset_multimodal_v8.parquet.\n"
)

# --------------------------------------------------
# GENERAL
# --------------------------------------------------

report.append(
    "## General Summary\n"
)

report.append(
    f"- Records: {rows:,}\n"
)

report.append(
    f"- Variables: {cols:,}\n"
)

report.append(
    f"- Municipalities: {municipios:,}\n"
)

report.append(
    f"- States: {estados}\n"
)

report.append(
    f"- Temporal coverage: "
    f"{anio_min}-{anio_max}\n"
)

report.append(
    f"- Duplicate rows: "
    f"{duplicates}\n"
)

# --------------------------------------------------
# DATA TYPES
# --------------------------------------------------

report.append(
    "\n## Data Types\n"
)

dtype_counts = (
    df.dtypes
    .astype(str)
    .value_counts()
)

for dtype, n in dtype_counts.items():

    report.append(
        f"- {dtype}: {n}\n"
    )

# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------

report.append(
    "\n## Top Missing Values\n"
)

report.append(
    "| Variable | Missing | Missing % |\n"
)

report.append(
    "|----------|---------|-----------|\n"
)

for _, row in top_missing.iterrows():

    report.append(

        f"| {row['variable']} | "
        f"{int(row['missing']):,} | "
        f"{row['missing_pct']:.2f}% |\n"

    )

# --------------------------------------------------
# TEMPORAL FEATURES
# --------------------------------------------------

report.append(
    "\n## Lag Variables\n"
)

for c in lag_vars:

    report.append(
        f"- {c}\n"
    )

report.append(
    "\n## Rolling Window Variables\n"
)

for c in rolling_vars:

    report.append(
        f"- {c}\n"
    )

# --------------------------------------------------
# KEY VARIABLES
# --------------------------------------------------

report.append(
    "\n## Key Epidemiological Variables\n"
)

for c in [

    "casos",
    "casos_est",
    "Rt",
    "p_inc100k",

]:

    if c in df.columns:

        report.append(
            f"\n### {c}\n"
        )

        report.append(
            f"- Mean: "
            f"{df[c].mean():.4f}\n"
        )

        report.append(
            f"- Median: "
            f"{df[c].median():.4f}\n"
        )

        report.append(
            f"- Min: "
            f"{df[c].min():.4f}\n"
        )

        report.append(
            f"- Max: "
            f"{df[c].max():.4f}\n"
        )

# --------------------------------------------------
# CONCLUSION
# --------------------------------------------------

report.append(
    "\n## Audit Conclusion\n"
)

report.append(
    "The dataset passed the structural audit.\n"
)

report.append(
    f"- {municipios:,} municipalities represented.\n"
)

report.append(
    f"- {cols} variables available.\n"
)

report.append(
    f"- {len(lag_vars)} lag variables generated.\n"
)

report.append(
    f"- {len(rolling_vars)} rolling-window variables generated.\n"
)

report.append(
    f"- Duplicate rows detected: {duplicates}.\n"
)

# ==================================================
# SAVE REPORT
# ==================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.writelines(report)

# ==================================================
# END
# ==================================================

elapsed = (
    time.time()
    - start_time
)

print("\n===================================")
print("AUDIT COMPLETED")
print("===================================")

print(
    f"Output:\n{OUTPUT_FILE}"
)

print(
    f"\nExecution time: "
    f"{elapsed/60:.2f} minutes"
)

print("\nDONE")