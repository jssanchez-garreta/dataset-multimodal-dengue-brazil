# ============================================================
# 34_benchmark_ridge.py
#
# Early Detection of Dengue Outbreaks in Brazil
#
# Ridge Linear Regression with L2 Regularization Benchmark
#
# Changes:
# - Median imputation within fold
# - StandardScaler within fold
# - Ridge Regression (Scikit-Learn)
#
# Dataset:
# dataset_benchmark_v1.parquet
#
# Target:
# p_inc100k
# ============================================================

import os
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import Ridge

# ==================================================
# REPRODUCIBILITY
# ==================================================

SEED = 42
np.random.seed(SEED)

# ==================================================
# CONFIGURATION
# ==================================================

TARGET = "p_inc100k"
ALPHA = 1.0  # Parámetro de regularización L2

# ==================================================
# TIMER
# ==================================================

start_time = time.time()

# ==================================================
# PATHS
# ==================================================

PROJECT_DIR = "/content/drive/MyDrive/DENGUE_BRASIL"

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
    "RIDGE_FORECASTING"
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
    "ridge_results.xlsx"
)

COEFFICIENTS_CSV = os.path.join(
    MODEL_DIR,
    "ridge_coefficients.csv"
)

PREDICTIONS_CSV = os.path.join(
    MODEL_DIR,
    "ridge_predictions.csv"
)

COEFFICIENTS_PNG = os.path.join(
    MODEL_DIR,
    "ridge_coefficients.png"
)

REPORT_MD = os.path.join(
    MODEL_DIR,
    "RIDGE_REPORT.md"
)

# ==================================================
# LOAD DATASET
# ==================================================

print("\n===================================")
print("LOADING DATASET")
print("===================================")

df = pd.read_parquet(DATASET_FILE)

print(df.shape)

# ==================================================
# FEATURES
# ==================================================

exclude_cols = [
    # identifiers
    "codigo_ibge",
    "municipio",
    "estado",

    # temporal identifiers
    "data_iniSE",
    "SE",
    "anio",
    "semana",

    # target
    TARGET,

    # leakage variables
    "casos_est",
    "casos_est_min",
    "casos_est_max",
    "casprov",
    "nivel_inc",
    "nivel",

    # current-week epidemiology
    "casos",
    "Rt",
    "p_rt1",
    "transmissao",
    "receptivo",
    "notif_accum_year"
]

features = [
    c for c in df.columns
    if c not in exclude_cols
]

X = df[features]

numeric_features = (
    X.select_dtypes(include=np.number)
    .columns
    .tolist()
)

features = numeric_features

X = X[features].astype(np.float32)
y = df[TARGET].astype(np.float32)

print(f"\nNumeric features: {len(features)}")

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
coefficients_all = []

# ==================================================
# TRAIN LOOP
# ==================================================

for fold_id, (train_end, test_year) in enumerate(folds, start=1):

    print("\n===================================")
    print(f"FOLD {fold_id}")
    print("===================================")

    train_mask = df["anio"] <= train_end
    test_mask = df["anio"] == test_year

    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]

    X_test = X.loc[test_mask]
    y_test = y.loc[test_mask]

    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows : {len(X_test):,}")

    # ==========================================
    # IMPUTATION
    # ==========================================

    imputer = SimpleImputer(strategy="median")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=features,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=features,
        index=X_test.index
    )

    # ==========================================
    # SCALING
    # ==========================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # MODEL
    # ==========================================

    model = Ridge(
        alpha=ALPHA,
        random_state=SEED
    )

    model.fit(X_train_scaled, y_train)

    # ==========================================
    # PREDICTIONS
    # ==========================================

    preds = model.predict(X_test_scaled)

    # ==========================================
    # METRICS
    # ==========================================

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2  : {r2:.4f}")

    metrics.append([
        fold_id,
        train_end,
        test_year,
        mae,
        rmse,
        r2
    ])

    pred_df = pd.DataFrame({
        "codigo_ibge": df.loc[test_mask, "codigo_ibge"],
        "anio": df.loc[test_mask, "anio"],
        "semana": df.loc[test_mask, "semana"],
        "real": y_test.values,
        "pred": preds
    })

    predictions_all.append(pred_df)

    # ==========================================
    # COEFFICIENTS / FEATURE IMPORTANCE
    # ==========================================

    coef_df = pd.DataFrame({
        "feature": features,
        "coefficient": model.coef_,
        "abs_coefficient": np.abs(model.coef_)
    })
    coef_df["fold"] = fold_id
    coefficients_all.append(coef_df)

# ==================================================
# METRICS TABLES
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
# COEFFICIENTS SUMMARY & FIGURE
# ==================================================

coefficients_df = pd.concat(coefficients_all)

coefficients_summary = (
    coefficients_df
    .groupby("feature")[["coefficient", "abs_coefficient"]]
    .mean()
    .sort_values(by="abs_coefficient", ascending=False)
    .reset_index()
)

coefficients_summary.to_csv(COEFFICIENTS_CSV, index=False)

top_features = coefficients_summary.head(20)

plt.figure(figsize=(10, 8))
sns.barplot(
    data=top_features,
    x="abs_coefficient",
    y="feature"
)
plt.title("Ridge Regression Top 20 Absolute Coefficients")
plt.xlabel("Mean Absolute Coefficient")
plt.tight_layout()
plt.savefig(COEFFICIENTS_PNG, dpi=300)
plt.close()

# ==================================================
# PREDICTIONS
# ==================================================

predictions_df = pd.concat(predictions_all)
predictions_df.to_csv(PREDICTIONS_CSV, index=False)

# ==================================================
# EXCEL
# ==================================================

with pd.ExcelWriter(RESULTS_XLSX) as writer:
    metrics_df.to_excel(writer, sheet_name="FOLD_METRICS", index=False)
    summary_df.to_excel(writer, sheet_name="SUMMARY", index=False)
    coefficients_summary.to_excel(writer, sheet_name="COEFFICIENTS", index=False)

# ==================================================
# REPORT
# ==================================================

report = f"""
# RIDGE FORECASTING REPORT

Target:
{TARGET}

Features:
{len(features)}

Model:
Ridge Regression (L2 Regularization)

Alpha: {ALPHA}

Mean Performance

MAE:
{summary_df.iloc[0]['Mean']:.4f} ± {summary_df.iloc[0]['Std']:.4f}

RMSE:
{summary_df.iloc[1]['Mean']:.4f} ± {summary_df.iloc[1]['Std']:.4f}

R2:
{summary_df.iloc[2]['Mean']:.4f} ± {summary_df.iloc[2]['Std']:.4f}
"""

with open(REPORT_MD, "w", encoding="utf-8") as f:
    f.write(report)

# ==================================================
# FINAL
# ==================================================

elapsed = time.time() - start_time

print("\n===================================")
print("RIDGE COMPLETED")
print("===================================")

print(f"\nMean MAE : {metrics_df['MAE'].mean():.4f}")
print(f"Mean RMSE: {metrics_df['RMSE'].mean():.4f}")
print(f"Mean R2  : {metrics_df['R2'].mean():.4f}")

print(f"\nResults:\n{MODEL_DIR}")
print(f"\nExecution time: {elapsed/60:.2f} minutes")

print("\nDONE")