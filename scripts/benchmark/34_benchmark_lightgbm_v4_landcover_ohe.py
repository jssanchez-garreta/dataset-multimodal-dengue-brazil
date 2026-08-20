# ============================================================
# 34_benchmark_lightgbm_v4_landcover_ohe.py
#
# LightGBM Forecasting
# + dominant_landcover One-Hot Encoding
# ============================================================

!pip install lightgbm -q

import os
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from lightgbm import LGBMRegressor

# ==================================================
# CONFIG
# ==================================================

TARGET = "p_inc100k"

N_ESTIMATORS = 200

LEARNING_RATE = 0.05

NUM_LEAVES = 15

MAX_BIN = 127

np.random.seed(42)

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
    "LIGHTGBM_V4_LANDCOVER_OHE"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

# ==================================================
# LOAD
# ==================================================

print("\n===================================")
print("LOADING DATASET")
print("===================================")

df = pd.read_parquet(
    DATASET_FILE
)

print(df.shape)

# ==================================================
# EXCLUSIONS
# ==================================================

exclude_cols = [

    "codigo_ibge",
    "municipio",
    "estado",

    "data_iniSE",
    "SE",

    TARGET,

    "casos_est",
    "casos_est_min",
    "casos_est_max",

    "casprov",

    "nivel_inc",
    "nivel",

    "casos",

    "Rt",

    "p_rt1",

    "transmissao",

    "receptivo",

    "notif_accum_year"

]

# ==================================================
# OHE LANDCOVER
# ==================================================

landcover_ohe = pd.get_dummies(

    df["dominant_landcover"],

    prefix="landcover",

    dtype=np.uint8

)

# ==================================================
# FEATURES
# ==================================================

features_df = df.drop(
    columns=exclude_cols,
    errors="ignore"
)

features_df = pd.concat(
    [
        features_df,
        landcover_ohe
    ],
    axis=1
)

features_df = features_df.drop(
    columns=[
        "dominant_landcover"
    ],
    errors="ignore"
)

numeric_cols = (

    features_df

    .select_dtypes(
        include=np.number
    )

    .columns

    .tolist()

)

X = (
    features_df[numeric_cols]
    .astype(np.float32)
)

y = (
    df[TARGET]
    .astype(np.float32)
)

print(
    f"\nNumeric features: "
    f"{len(numeric_cols)}"
)

# ==================================================
# FOLDS
# ==================================================

folds = [

    (2020, 2021),
    (2021, 2022),
    (2022, 2023),
    (2023, 2024),
    (2024, 2025)

]

metrics = []

feature_importances = []

# ==================================================
# LOOP
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

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X.columns,
        index=X_train.index
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=X.columns,
        index=X_test.index
    )

    model = LGBMRegressor(

        objective="regression",

        n_estimators=N_ESTIMATORS,

        learning_rate=LEARNING_RATE,

        num_leaves=NUM_LEAVES,

        max_bin=MAX_BIN,

        subsample=0.8,

        colsample_bytree=0.8,

        random_state=42,

        n_jobs=-1

    )

    model.fit(
        X_train,
        y_train
    )

    preds = model.predict(
        X_test
    )

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
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R2   : {r2:.4f}"
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

    fi = pd.DataFrame({

        "feature":
            X.columns,

        "importance":
            model.feature_importances_

    })

    fi["fold"] = fold_id

    feature_importances.append(
        fi
    )

# ==================================================
# RESULTS
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

feature_importance_df = (

    pd.concat(feature_importances)

    .groupby("feature")["importance"]

    .mean()

    .sort_values(
        ascending=False
    )

    .reset_index()

)

# ==================================================
# SAVE
# ==================================================

metrics_df.to_csv(

    os.path.join(
        MODEL_DIR,
        "fold_metrics.csv"
    ),

    index=False

)

summary_df.to_csv(

    os.path.join(
        MODEL_DIR,
        "summary_metrics.csv"
    ),

    index=False

)

feature_importance_df.to_csv(

    os.path.join(
        MODEL_DIR,
        "feature_importance.csv"
    ),

    index=False

)

# ==================================================
# FINAL
# ==================================================

print("\n===================================")
print("LIGHTGBM V4 COMPLETED")
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
    f"{(time.time()-start_time)/60:.2f} minutes"
)

print("\nDONE")