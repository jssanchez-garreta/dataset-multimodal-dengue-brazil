# ============================================================
# 32_benchmark_random_forest.py
#
# Early Detection of Dengue Outbreaks in Brazil
#
# Random Forest Benchmark
#
# Dataset:
# dataset_benchmark_v1.parquet
#
# Target:
# p_inc100k
#
# Forecasting configuration:
#
# n_estimators = 50
# max_depth = 10
# min_samples_leaf = 20
# max_features = "sqrt"
# ============================================================

import os
import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

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

N_ESTIMATORS = 50

MAX_DEPTH = 10

MIN_SAMPLES_LEAF = 20

MAX_FEATURES = "sqrt"

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
    "RANDOM_FOREST_V3_FORECASTING"
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
    "rf_results.xlsx"
)

IMPORTANCE_CSV = os.path.join(
    MODEL_DIR,
    "rf_feature_importance.csv"
)

PREDICTIONS_CSV = os.path.join(
    MODEL_DIR,
    "rf_predictions.csv"
)

IMPORTANCE_PNG = os.path.join(
    MODEL_DIR,
    "rf_feature_importance.png"
)

REPORT_MD = os.path.join(
    MODEL_DIR,
    "RF_REPORT.md"
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

    # target

    TARGET,

    # leakage

    "casos_est",
    "casos_est_min",
    "casos_est_max",

    "casprov",

    "nivel_inc",

    "nivel",

    # current week epidemiology

    "casos",

    "Rt",

    "p_rt1",

    "transmissao",

    "receptivo",

    "notif_accum_year"

]

features = [

    c

    for c in df.columns

    if c not in exclude_cols

]

X = df[features]

numeric_features = (

    X.select_dtypes(
        include=np.number
    )

    .columns

    .tolist()

)

features = numeric_features

X = (
    X[features]
    .astype(np.float32)
)

y = (
    df[TARGET]
    .astype(np.float32)
)

print(
    f"\nNumeric features: {len(features)}"
)

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

feature_importances = []

# ==================================================
# TRAIN LOOP
# ==================================================

for fold_id, (train_end, test_year) in enumerate(
    folds,
    start=1
):

    print("\n===================================")
    print(f"FOLD {fold_id}")
    print("===================================")

    train_mask = (
        df["anio"] <= train_end
    )

    test_mask = (
        df["anio"] == test_year
    )

    X_train = X.loc[
        train_mask
    ]

    y_train = y.loc[
        train_mask
    ]

    X_test = X.loc[
        test_mask
    ]

    y_test = y.loc[
        test_mask
    ]

    print(
        f"Train rows: {len(X_train):,}"
    )

    print(
        f"Test rows : {len(X_test):,}"
    )

    # ------------------------------------------
    # IMPUTATION
    # ------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

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

    # ------------------------------------------
    # MODEL
    # ------------------------------------------

    model = RandomForestRegressor(

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        min_samples_leaf=MIN_SAMPLES_LEAF,

        max_features=MAX_FEATURES,

        n_jobs=-1,

        random_state=42

    )

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(
        X_test
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

    # ------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------

    fi = pd.DataFrame({

        "feature":
            features,

        "importance":
            model.feature_importances_

    })

    fi["fold"] = fold_id

    feature_importances.append(
        fi
    )

# ==================================================
# METRICS
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
# FEATURE IMPORTANCE
# ==================================================

feature_importance_df = pd.concat(
    feature_importances
)

feature_importance_df = (

    feature_importance_df

    .groupby("feature")["importance"]

    .mean()

    .sort_values(
        ascending=False
    )

    .reset_index()

)

feature_importance_df.to_csv(
    IMPORTANCE_CSV,
    index=False
)

# ==================================================
# FEATURE IMPORTANCE FIGURE
# ==================================================

top_features = (
    feature_importance_df
    .head(20)
)

plt.figure(
    figsize=(10, 8)
)

sns.barplot(
    data=top_features,
    x="importance",
    y="feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    IMPORTANCE_PNG,
    dpi=300
)

plt.close()

# ==================================================
# PREDICTIONS
# ==================================================

predictions_df = pd.concat(
    predictions_all
)

predictions_df.to_csv(
    PREDICTIONS_CSV,
    index=False
)

# ==================================================
# EXCEL
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

    feature_importance_df.to_excel(
        writer,
        sheet_name="FEATURE_IMPORTANCE",
        index=False
    )

# ==================================================
# REPORT
# ==================================================

report = f"""
# RANDOM FOREST V3 FORECASTING

Target:
{TARGET}

Features:
{len(features)}

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
print("RANDOM FOREST COMPLETED")
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