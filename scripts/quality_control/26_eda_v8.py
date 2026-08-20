# ============================================================
# 26_eda_v8.py
#
# PROYECTO:
# Early Detection of Dengue Outbreaks in Brazil
#
# OBJETIVO:
#
# Exploratory Data Analysis (EDA)
#
# ENTRADA:
#
# dataset_multimodal_v8.parquet
#
# SALIDAS:
#
# EDA_V8/
#
# ├── FIGURES/
# │   ├── missing_values.png
# │   ├── dengue_timeseries_brazil.png
# │   ├── climate_distributions.png
# │   ├── sanitation_distributions.png
# │   ├── demographic_distributions.png
# │   ├── lag_distributions.png
# │   └── correlation_matrix.png
# │
# ├── TABLES_CSV/
# │   ├── dataset_overview.csv
# │   ├── missing_values.csv
# │   ├── correlation_ranking.csv
# │   └── feature_groups.csv
# │
# └── EDA_V8_REPORT.md
#
# ============================================================

# ==================================================
# IMPORTS
# ==================================================

import os
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

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

EDA_DIR = os.path.join(
    DATASET_DIR,
    "EDA_V8"
)

FIGURES_DIR = os.path.join(
    EDA_DIR,
    "FIGURES"
)

TABLES_DIR = os.path.join(
    EDA_DIR,
    "TABLES_CSV"
)

os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ==================================================
# LOAD DATASET
# ==================================================

print("\n===================================")
print("LOADING DATASET")
print("===================================")

df = pd.read_parquet(INPUT_FILE)

print(df.shape)

# ==================================================
# BASIC METRICS
# ==================================================

rows = len(df)

cols = len(df.columns)

municipios = df["codigo_ibge"].nunique()

estados = df["estado"].nunique()

anio_min = int(df["anio"].min())

anio_max = int(df["anio"].max())

duplicates = int(df.duplicated().sum())

# ==================================================
# DATASET OVERVIEW TABLE
# ==================================================

overview = pd.DataFrame(
    {
        "Metric": [
            "Records",
            "Variables",
            "Municipalities",
            "States",
            "Temporal_Start",
            "Temporal_End",
            "Duplicate_Rows",
        ],
        "Value": [
            rows,
            cols,
            municipios,
            estados,
            anio_min,
            anio_max,
            duplicates,
        ],
    }
)

overview.to_csv(
    os.path.join(
        TABLES_DIR,
        "dataset_overview.csv"
    ),
    index=False
)

# ==================================================
# MISSING VALUES
# ==================================================

print("\n===================================")
print("MISSING VALUES")
print("===================================")

missing = pd.DataFrame(
    {
        "variable": df.columns,
        "missing": df.isna().sum()
    }
)

missing["missing_pct"] = (
    missing["missing"] / rows * 100
)

missing = missing.sort_values(
    "missing",
    ascending=False
)

missing.to_csv(
    os.path.join(
        TABLES_DIR,
        "missing_values.csv"
    ),
    index=False
)

plt.figure(
    figsize=(10, 8)
)

missing.head(20).sort_values(
    "missing"
).plot(
    x="variable",
    y="missing",
    kind="barh",
    legend=False
)

plt.title(
    "Top 20 Missing Variables"
)

plt.xlabel(
    "Missing Records"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "missing_values.png"
    ),
    dpi=300
)

plt.close()

# ==================================================
# NATIONAL DENGUE TIME SERIES
# ==================================================

print("\n===================================")
print("DENGUE TIME SERIES")
print("===================================")

national_cases = (
    df.groupby(
        ["anio", "semana"]
    )["casos"]
    .sum()
    .reset_index()
)

national_cases["idx"] = np.arange(
    len(national_cases)
)

plt.figure(
    figsize=(14, 5)
)

plt.plot(
    national_cases["idx"],
    national_cases["casos"],
    linewidth=1
)

plt.title(
    "Weekly Dengue Cases in Brazil"
)

plt.xlabel(
    "Weeks"
)

plt.ylabel(
    "Cases"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "dengue_timeseries_brazil.png"
    ),
    dpi=300
)

plt.close()

# ==================================================
# CLIMATE DISTRIBUTIONS
# ==================================================

print("\n===================================")
print("CLIMATE DISTRIBUTIONS")
print("===================================")

climate_vars = [
    "tempmed",
    "umidmed",
    "precip_total_semana",
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)

for ax, var in zip(
    axes,
    climate_vars
):

    sns.histplot(
        df[var],
        bins=50,
        ax=ax
    )

    ax.set_title(var)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "climate_distributions.png"
    ),
    dpi=300
)

plt.close()

# ==================================================
# SANITATION DISTRIBUTIONS
# ==================================================

print("\n===================================")
print("SANITATION")
print("===================================")

san_vars = [
    "water_supply_pct",
    "sewage_pct",
    "garbage_collection_pct",
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)

for ax, var in zip(
    axes,
    san_vars
):

    sns.histplot(
        df[var],
        bins=50,
        ax=ax
    )

    ax.set_title(var)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "sanitation_distributions.png"
    ),
    dpi=300
)

plt.close()

# ==================================================
# DEMOGRAPHIC DISTRIBUTIONS
# ==================================================

print("\n===================================")
print("DEMOGRAPHY")
print("===================================")

demo_vars = [
    "population_density",
    "median_age",
    "aging_index",
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)

for ax, var in zip(
    axes,
    demo_vars
):

    sns.histplot(
        df[var],
        bins=50,
        ax=ax
    )

    ax.set_title(var)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "demographic_distributions.png"
    ),
    dpi=300
)

plt.close()

# ==================================================
# TEMPORAL FEATURES
# ==================================================

print("\n===================================")
print("TEMPORAL FEATURES")
print("===================================")

lag_vars = [
    "casos_lag_1",
    "casos_lag_4",
    "casos_roll4_mean",
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)

for ax, var in zip(
    axes,
    lag_vars
):

    sns.histplot(
        df[var],
        bins=50,
        ax=ax
    )

    ax.set_title(var)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "lag_distributions.png"
    ),
    dpi=300
)

plt.close()

# ==================================================
# FEATURE GROUPS TABLE
# ==================================================

feature_groups = pd.DataFrame(
    {
        "Group": [
            "Epidemiology",
            "Meteorology",
            "Precipitation",
            "MODIS",
            "ENSO",
            "Topography",
            "Land Cover",
            "Demography",
            "Sanitation",
            "Temporal Features",
        ],
        "Description": [
            "Dengue indicators",
            "Temperature and humidity",
            "CHIRPS precipitation",
            "Vegetation and LST",
            "Global climate indicators",
            "Elevation metrics",
            "Land cover percentages",
            "Population structure",
            "Water, sewage and waste",
            "Lag and rolling features",
        ],
    }
)

feature_groups.to_csv(
    os.path.join(
        TABLES_DIR,
        "feature_groups.csv"
    ),
    index=False
)

# ==================================================
# CORRELATION ANALYSIS
# ==================================================

print("\n===================================")
print("CORRELATION ANALYSIS")
print("===================================")

# --------------------------------------------------
# CORRELATION WITH TARGET (casos)
# --------------------------------------------------

corr_target = []

numeric_cols = df.select_dtypes(
    include=np.number
).columns

for col in numeric_cols:

    if col == "casos":
        continue

    try:

        value = (
            df[
                ["casos", col]
            ]
            .corr()
            .iloc[0, 1]
        )

        corr_target.append(
            [col, abs(value)]
        )

    except Exception:
        pass

corr_target = pd.DataFrame(
    corr_target,
    columns=[
        "Variable",
        "AbsoluteCorrelation"
    ]
)

corr_target = corr_target.sort_values(
    "AbsoluteCorrelation",
    ascending=False
)

# --------------------------------------------------
# SAVE COMPLETE RANKING
# --------------------------------------------------

corr_target.to_csv(
    os.path.join(
        TABLES_DIR,
        "all_correlations.csv"
    ),
    index=False
)

# --------------------------------------------------
# SAVE TOP-RANKING TABLE
# --------------------------------------------------

corr_target.head(30).to_csv(
    os.path.join(
        TABLES_DIR,
        "correlation_ranking.csv"
    ),
    index=False
)

# --------------------------------------------------
# TOP VARIABLES FOR HEATMAP
# --------------------------------------------------

top_vars = (
    corr_target
    .head(30)
    ["Variable"]
    .tolist()
)

corr_vars = (
    ["casos"]
    + top_vars
)

corr_df = df[corr_vars]

corr_matrix = corr_df.corr(
    numeric_only=True
)

# --------------------------------------------------
# HEATMAP
# --------------------------------------------------

plt.figure(
    figsize=(16, 14)
)

sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    center=0
)

plt.title(
    "Top 30 Variables Correlated with Dengue Cases"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURES_DIR,
        "correlation_matrix.png"
    ),
    dpi=300
)

plt.close()

print(
    f"Total numeric variables analysed: "
    f"{len(numeric_cols)}"
)

print(
    f"Variables shown in heatmap: "
    f"{len(corr_vars)}"
)

print("\nTOP 10 CORRELATIONS")

print(
    corr_target.head(10)
)

# ==================================================
# MARKDOWN REPORT
# ==================================================

report = f"""
# EDA_V8_REPORT

## Dataset Summary

- Records: {rows:,}
- Variables: {cols}
- Municipalities: {municipios:,}
- States: {estados}
- Temporal Coverage: {anio_min}-{anio_max}
- Duplicate Rows: {duplicates}

## Generated Figures

- missing_values.png
- dengue_timeseries_brazil.png
- climate_distributions.png
- sanitation_distributions.png
- demographic_distributions.png
- lag_distributions.png
- correlation_matrix.png

## Generated CSV Tables

- dataset_overview.csv
- missing_values.csv
- correlation_ranking.csv  -> top 30 correlations
- all_correlations.csv     -> complete ranking
- feature_groups.csv

## Preliminary Assessment

The dataset appears structurally consistent
and suitable for benchmark machine learning
experiments.

Important missing values are concentrated in:

- sewage_pct
- sanitation variables
- LST variables
- humidity and temperature variables

Temporal feature engineering was successfully
integrated through lag and rolling features.
"""

with open(
    os.path.join(
        EDA_DIR,
        "EDA_V8_REPORT.md"
    ),
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
print("EDA COMPLETED")
print("===================================")

print(
    f"Output folder:\n{EDA_DIR}"
)

print(
    f"\nExecution time: "
    f"{elapsed/60:.2f} minutes"
)

print("\nDONE")