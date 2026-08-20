# ============================================================
# 28_build_benchmark_dataset.py
#
# PROYECTO:
# Early Detection of Dengue Outbreaks in Brazil
#
# OBJETIVO:
#
# Crear un dataset específico para benchmarking
# a partir de DATASET_MULTIMODAL_V8.
#
# SALIDA:
#
# dataset_benchmark_v1.parquet
# benchmark_dataset_report.md
#
# ============================================================

import os
import time

import pandas as pd

# ==================================================
# CONFIG
# ==================================================

SAMPLE_PER_YEAR = 50_000

RANDOM_STATE = 42

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

INPUT_FILE = os.path.join(
    PROJECT_DIR,
    "DATASETS_MASTER_MULTIMODAL",
    "dataset_multimodal_v8.parquet"
)

BENCHMARK_DIR = os.path.join(
    PROJECT_DIR,
    "BENCHMARK"
)

os.makedirs(
    BENCHMARK_DIR,
    exist_ok=True
)

OUTPUT_FILE = os.path.join(
    BENCHMARK_DIR,
    "dataset_benchmark_v1.parquet"
)

REPORT_FILE = os.path.join(
    BENCHMARK_DIR,
    "benchmark_dataset_report.md"
)

# ==================================================
# LOAD
# ==================================================

print("\n===================================")
print("LOADING DATASET")
print("===================================")

df = pd.read_parquet(
    INPUT_FILE
)

print(df.shape)

# ==================================================
# TEMPORAL STRATIFIED SAMPLING
# ==================================================

print("\n===================================")
print("TEMPORAL SAMPLING")
print("===================================")

samples = []

years = sorted(
    df["anio"].unique()
)

for year in years:

    subset = df[
        df["anio"] == year
    ]

    n_available = len(subset)

    n_sample = min(
        SAMPLE_PER_YEAR,
        n_available
    )

    sampled = subset.sample(
        n=n_sample,
        random_state=RANDOM_STATE
    )

    samples.append(
        sampled
    )

    print(
        f"{year}: "
        f"{n_sample:,} rows"
    )

# ==================================================
# MERGE
# ==================================================

benchmark_df = pd.concat(
    samples,
    ignore_index=True
)

# ==================================================
# SORT
# ==================================================

benchmark_df = benchmark_df.sort_values(
    [
        "anio",
        "semana",
        "codigo_ibge"
    ]
)

# ==================================================
# STATS
# ==================================================

rows = len(benchmark_df)

cols = len(benchmark_df.columns)

municipios = (
    benchmark_df["codigo_ibge"]
    .nunique()
)

estados = (
    benchmark_df["estado"]
    .nunique()
)

duplicates = (
    benchmark_df
    .duplicated()
    .sum()
)

# ==================================================
# EXPORT DATASET
# ==================================================

print("\n===================================")
print("EXPORT")
print("===================================")

benchmark_df.to_parquet(
    OUTPUT_FILE,
    index=False
)

# ==================================================
# REPORT
# ==================================================

report = f"""
# DATASET_BENCHMARK_V1

Source dataset:
DATASET_MULTIMODAL_V8

Sampling strategy:
Temporal stratified sampling by year

Rows sampled per year:
{SAMPLE_PER_YEAR:,}

Dataset summary

Records:
{rows:,}

Variables:
{cols}

Municipalities:
{municipios:,}

States:
{estados}

Duplicate rows:
{duplicates}

Temporal coverage:
{benchmark_df['anio'].min()}-{benchmark_df['anio'].max()}

This dataset was created to support
benchmark experiments using:

- Random Forest
- LightGBM
- XGBoost
- CatBoost
"""

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

# ==================================================
# FINAL
# ==================================================

elapsed = (
    time.time()
    - start_time
)

print("\n===================================")
print("DONE")
print("===================================")

print(
    f"\nOutput dataset:\n{OUTPUT_FILE}"
)

print(
    f"\nRows: {rows:,}"
)

print(
    f"Columns: {cols}"
)

print(
    f"Municipalities: {municipios:,}"
)

print(
    f"\nExecution time: "
    f"{elapsed/60:.2f} minutes"
)