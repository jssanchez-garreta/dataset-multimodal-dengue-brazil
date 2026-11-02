# ============================================================
# 33_benchmark_deep_forest_simplified.py
#
# Early Detection of Dengue Outbreaks in Brazil
#
# Simplified Deep Forest Benchmark (Cascade Forest Inspired Benchmark)
#
# Dataset:
# dataset_benchmark_v1.parquet
#
# Target:
# p_inc100k
#
# Forecasting configuration:
#
# Benchmark dataset (800,000 rows)
# Elimination of variables with targeted leakage
# Elimination of current-week epidemiology
# Only numeric variables
# float32
# Imputation by median within each fold
# Time validation with expanding window
# Excel, CSV, and Markdown report
# Separate folder: DENGUE_BRASIL/BENCHMARK/DEEP_FOREST_FORECASTING
# ============================================================

import os
import time

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)

# ==================================================
# SIMPLE DEEP FOREST REGRESSOR
# Simplified gcForest-inspired implementation
# ==================================================

class SimpleDeepForestRegressor:

    def __init__(

        self,

        n_layers=2,

        n_estimators_rf=30,

        n_estimators_et=30,

        random_state=42

    ):

        self.n_layers = n_layers

        self.n_estimators_rf = n_estimators_rf

        self.n_estimators_et = n_estimators_et

        self.random_state = random_state

        self.layers = []

    # ------------------------------------------------

    def fit(self, X, y):

        X_current = np.asarray(
            X,
            dtype=np.float32
        )

        y = np.asarray(
            y,
            dtype=np.float32
        )

        self.layers = []

        for layer_id in range(
            self.n_layers
        ):

            layer_start = time.time()

            print(
                f"  Layer {layer_id + 1}/"
                f"{self.n_layers}"
            )

            rf = RandomForestRegressor(

                n_estimators=
                self.n_estimators_rf,

                n_jobs=-1,

                random_state=
                self.random_state
                + layer_id
            )

            et = ExtraTreesRegressor(

                n_estimators=
                self.n_estimators_et,

                n_jobs=-1,

                random_state=
                self.random_state
                + 100
                + layer_id
            )

            # ------------------------------------
            # FIT FULL MODELS
            # ------------------------------------

            print("TRAINING RF")

            rf.fit(
                X_current,
                y
            )

            print("TRAINING ET")

            et.fit(
                X_current,
                y
            )

            print("GENERATING CASCADE FEATURES")

            rf_pred = rf.predict(
                X_current
            ).reshape(-1, 1)

            et_pred = et.predict(
                X_current
            ).reshape(-1, 1)

            X_current = np.hstack([

                X_current,

                rf_pred,

                et_pred

            ]).astype(
                np.float32
            )

            self.layers.append(
                (rf, et)
            )

            print(
                f"Layer {layer_id + 1} completed ",
                f"({(time.time()-layer_start)/60:.2f} min)"
            )

        return self

    # ------------------------------------------------

    def predict(self, X):

        X_current = np.asarray(
            X,
            dtype=np.float32
        )

        rf_pred = None

        et_pred = None

        for rf, et in self.layers:

            rf_pred = rf.predict(
                X_current
            ).reshape(-1, 1)

            et_pred = et.predict(
                X_current
            ).reshape(-1, 1)

            X_current = np.hstack([

                X_current,

                rf_pred,

                et_pred

            ]).astype(
                np.float32
            )

        return (

            rf_pred.ravel()

            +

            et_pred.ravel()

        ) / 2
             
# ==================================================
# REPRODUCIBILITY
# ==================================================

np.random.seed(42)

# ==================================================
# CONFIGURATION
# ==================================================

TARGET = "p_inc100k"

N_LAYERS = 2

N_ESTIMATORS_RF = 30

N_ESTIMATORS_ET = 30

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
    "CASCADE_FOREST_V1_FORECASTING"
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
    "cascade_forest_results.xlsx"
)

PREDICTIONS_CSV = os.path.join(
    MODEL_DIR,
    "cascade_forest_predictions.csv"
)

REPORT_MD = os.path.join(
    MODEL_DIR,
    "CASCADE_FOREST_REPORT.md"
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

print(
    pd.Series(features)
    .sort_values()
    .to_string(index=False)
)

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

    # ------------------------------------------
    # MODEL
    # ------------------------------------------

    model = SimpleDeepForestRegressor(

        n_layers=N_LAYERS,

        n_estimators_rf=N_ESTIMATORS_RF,

        n_estimators_et=N_ESTIMATORS_ET,

        random_state=42

    )

    model.fit(
        X_train.values,
        y_train.values
    )

    preds = model.predict(
        X_test.values
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
# DEEP FOREST FORECASTING

Layers:
{N_LAYERS}

RF Trees:
{N_ESTIMATORS_RF}

ET Trees:
{N_ESTIMATORS_ET}

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
print("CASCADE FOREST COMPLETED")
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