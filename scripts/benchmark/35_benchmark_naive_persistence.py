# ============================================================
# 35_benchmark_naive_persistence.py
#
# Early Detection of Dengue Outbreaks in Brazil
#
# Naïve Persistence Benchmark (Incidence_t = Incidence_{t-1})
#
# Dataset:
# dataset_benchmark_v1.parquet
#
# Target:
# p_inc100k
#
# Forecasting configuration:
# - Pred_t = p_inc100k_lag_1
# - No model training required
# - Time validation with expanding window
# - Excel, CSV, and Markdown report
# - Separate folder: DENGUE_BRASIL/BENCHMARK/NAIVE_PERSISTENCE_FORECASTING
# ============================================================

import os
import time

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==================================================
# REPRODUCIBILITY
# ==================================================

np.random.seed(42)

# ==================================================
# CONFIGURATION
# ==================================================

TARGET = "p_inc100k"
PERSISTENCE_FEATURE = "p_inc100k_lag_1"  # Si no existe, se usará casos_lag_1 / pop * 100k

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

DATASET_FILE = os.path.join(
    PROJECT_DIR,
    "BENCHMARK",
    "dataset_benchmark_v1.parquet"
)

BENCHMARK_DIR = os.path.join(
    PROJECT_DIR,
    "BENCHMARK"
)

MODEL_DIR = os.path.join(
    BENCHMARK_DIR,
    "NAIVE_PERSISTENCE_FORECASTING"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

# ==================================================
# OUTPUT FILES
# ==================================================

RESULTS_XLSX = os.path.join(
    MODEL_DIR,
    "naive_results.xlsx"
)

PREDICTIONS_CSV = os.path.join(
    MODEL_DIR,
    "naive_predictions.csv"
)

REPORT_MD = os.path.join(
    MODEL_DIR,
    "NAIVE_REPORT.md"
)

# ==================================================
# LOAD DATASET
# ==================================================

print("\n===================================")
print("LOADING DATASET")
print("===================================")

df = pd.read_parquet(
    DATASET_FILE
)

print(df.shape)

# ==================================================
# VERIFY PERSISTENCE FEATURE
# ==================================================

# Si p_inc100k_lag_1 no está directamente en las columnas, la calculamos usando casos_lag_1 y pop
if PERSISTENCE_FEATURE not in df.columns:
    print(f"\n'{PERSISTENCE_FEATURE}' not found in columns. Calculating from 'casos_lag_1' and 'pop'...")
    df[PERSISTENCE_FEATURE] = (df["casos_lag_1"] / df["pop"]) * 100000.0

y = df[TARGET].astype(np.float32)

# ==================================================
# TEMPORAL FOLDS
# ==================================================

folds = [
    (2020, 2021),
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025)
]

metrics = []
predictions_all = []

# ==================================================
# EVALUATION LOOP
# ==================================================

for fold_id, (train_end, test_year) in enumerate(
    folds,
    start=1
):

    print("\n===================================")
    print(f"FOLD {fold_id} (Test Year: {test_year})")
    print("===================================")

    test_mask = (
        df["anio"] == test_year
    )

    y_test = y.loc[test_mask]
    
    # La predicción es simplemente el valor persistente de la semana anterior (t-1)
    preds = df.loc[test_mask, PERSISTENCE_FEATURE].fillna(0.0).values

    print(
        f"Test rows : {len(y_test):,}"
    )

    # ------------------------------------------
    # METRICS
    # ------------------------------------------

    mae = mean_absolute_error(
        y_test,
        preds
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            preds
        )
    )

    r2 = r2_score(
        y_test,
        preds
    )

    print(
        f"MAE : {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"R2  : {r2:.4f}"
    )

    metrics.append(
        [
            fold_id,
            train_end,
            test_year,
            mae,
            rmse,
            r2
        ]
    )

    # ------------------------------------------
    # PREDICTIONS
    # ------------------------------------------

    pred_df = pd.DataFrame({

        "codigo_ibge":
            df.loc[test_mask, "codigo_ibge"],

        "anio":
            df.loc[test_mask, "anio"],

        "semana":
            df.loc[test_mask, "semana"],

        "real":
            y_test.values,

        "pred":
            preds

    })

    predictions_all.append(
        pred_df
    )

# ==================================================
# METRICS SUMMARY
# ==================================================

metrics_df = pd.DataFrame(

    metrics,

    columns=[
        "Fold",
        "Train_End",
        "Test_Year",
        "MAE",
        "RMSE",
        "R2"
    ]

)

summary_df = pd.DataFrame({

    "Metric": [
        "MAE",
        "RMSE",
        "R2"
    ],

    "Mean": [

        metrics_df["MAE"].mean(),

        metrics_df["RMSE"].mean(),

        metrics_df["R2"].mean()

    ],

    "Std": [

        metrics_df["MAE"].std(),

        metrics_df["RMSE"].std(),

        metrics_df["R2"].std()

    ]

})

# ==================================================
# PREDICTIONS EXPORT
# ==================================================

predictions_df = pd.concat(
    predictions_all
)

predictions_df.to_csv(
    PREDICTIONS_CSV,
    index=False
)

# ==================================================
# EXCEL EXPORT
# ==================================================

with pd.ExcelWriter(
    RESULTS_XLSX
) as writer:

    metrics_df.to_excel(
        writer,
        sheet_name="FOLD_METRICS",
        index=False
    )

    summary_df.to_excel(
        writer,
        sheet_name="SUMMARY",
        index=False
    )

# ==================================================
# REPORT
# ==================================================

report = f"""
# NAÏVE PERSISTENCE FORECASTING

Target:
{TARGET}

Persistence Feature:
{PERSISTENCE_FEATURE}

Mean Performance

MAE:
{summary_df.iloc[0]['Mean']:.4f}
±
{summary_df.iloc[0]['Std']:.4f}

RMSE:
{summary_df.iloc[1]['Mean']:.4f}
±
{summary_df.iloc[1]['Std']:.4f}

R2:
{summary_df.iloc[2]['Mean']:.4f}
±
{summary_df.iloc[2]['Std']:.4f}
"""

with open(
    REPORT_MD,
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
print("NAÏVE PERSISTENCE COMPLETED")
print("===================================")

print(
    f"\nMean MAE : {metrics_df['MAE'].mean():.4f}"
)

print(
    f"Mean RMSE: {metrics_df['RMSE'].mean():.4f}"
)

print(
    f"Mean R2  : {metrics_df['R2'].mean():.4f}"
)

print(
    f"\nResults:\n{MODEL_DIR}"
)

print(
    f"\nExecution time: "
    f"{elapsed/60:.2f} minutes"
)

print("\nDONE")