# ============================================================
# 37_benchmark_mlp.py
#
# Early Detection of Dengue Outbreaks in Brazil
#
# TensorFlow/Keras MLP Benchmark
#
# Changes:
# - Median imputation within fold
# - StandardScaler within fold
# - MLP instead of LightGBM
#
# Dataset:
# dataset_benchmark_v1.parquet
#
# Target:
# p_inc100k
# ============================================================

import os
import time
import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    BatchNormalization
)
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping

# ==================================================
# REPRODUCIBILITY
# ==================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ==================================================
# CONFIGURATION
# ==================================================

TARGET = "p_inc100k"

EPOCHS = 100
PATIENCE = 10
VALIDATION_SPLIT = 0.10
BATCH_SIZE = 4096
LEARNING_RATE = 0.001

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
    "MLP_V2_FORECASTING"
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
    "mlp_results.xlsx"
)

PREDICTIONS_CSV = os.path.join(
    MODEL_DIR,
    "mlp_predictions.csv"
)

REPORT_MD = os.path.join(
    MODEL_DIR,
    "MLP_REPORT.md"
)

TRAINING_CURVES_PNG = os.path.join(
    MODEL_DIR,
    "mlp_training_curves.png"
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

histories = []

# ==================================================
# MODEL FACTORY
# ==================================================

def build_mlp(n_features):

    model = Sequential([

        Dense(
            256,
            activation="relu",
            input_shape=(n_features,)
        ),

        BatchNormalization(),

        Dense(
            128,
            activation="relu"
        ),

        BatchNormalization(),

        Dense(
            64,
            activation="relu"
        ),

        Dense(1)

    ])

    model.compile(
        optimizer=Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="mse"
    )

    return model

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

    # ==========================================
    # IMPUTATION
    # ==========================================

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = pd.DataFrame(
        imputer.fit_transform(
            X_train
        ),
        columns=features,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        imputer.transform(
            X_test
        ),
        columns=features,
        index=X_test.index
    )

    # ==========================================
    # SCALING
    # ==========================================

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    # ==========================================
    # MODEL
    # ==========================================

    model = build_mlp(
        X_train.shape[1]
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        shuffle=True,
        callbacks=[early_stopping]
    )
    
    histories.append(history)

    # ==========================================
    # PREDICTIONS
    # ==========================================

    preds = (
        model.predict(
            X_test,
            batch_size=BATCH_SIZE,
            verbose=0
        )
        .ravel()
    )

    # ==========================================
    # METRICS
    # ==========================================

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

    metrics.append([
        fold_id,
        train_end,
        test_year,
        mae,
        rmse,
        r2
    ])

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
# TRAINING CURVES
# ==================================================

plt.figure(
    figsize=(10, 6)
)

for i, history in enumerate(
    histories,
    start=1
):

    plt.plot(
        history.history["loss"],
        label=f"Fold {i} Train"
    )

    plt.plot(
        history.history["val_loss"],
        linestyle="--",
        label=f"Fold {i} Val"
    )

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MLP Training and Validation Loss")
plt.legend()

plt.tight_layout()

plt.savefig(
    TRAINING_CURVES_PNG,
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

# ==================================================
# REPORT
# ==================================================

report = f"""
# MLP V2 FORECASTING

Target:
{TARGET}

Features:
{len(features)}

Architecture:
256 -> 128 -> 64 -> 1

Epochs:
{EPOCHS}

Batch size:
{BATCH_SIZE}

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
print("MLP V1 COMPLETED")
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