# ============================================================
# 33_compare_models.py
#
# Early Detection of Dengue Outbreaks in Brazil
#
# COMPARISON OF:
#
# 1) LIGHTGBM NOWCASTING
# 2) LIGHTGBM FORECASTING
# 3) XGBOOST FORECASTING
# 4) CATBOOST FORECASTING
# 5) RANDOM FOREST FORECASTING
#
# OUTPUT:
#
# BENCHMARK/COMPARISON/
#
# ============================================================

import os
import time

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon

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

BENCHMARK_DIR = os.path.join(
    PROJECT_DIR,
    "BENCHMARK"
)

COMPARISON_DIR = os.path.join(
    BENCHMARK_DIR,
    "COMPARISON"
)

os.makedirs(
    COMPARISON_DIR,
    exist_ok=True
)

# ==================================================
# FORECASTING MODELS
# ==================================================

FORECAST_MODELS = {

    "LightGBM":

        os.path.join(
            BENCHMARK_DIR,
            "LIGHTGBM_V3_FORECASTING",
            "lgbm_results.xlsx"
        ),

    "XGBoost":

        os.path.join(
            BENCHMARK_DIR,
            "XGBOOST_V3_FORECASTING",
            "xgb_results.xlsx"
        ),

    "CatBoost":

        os.path.join(
            BENCHMARK_DIR,
            "CATBOOST_V3_FORECASTING",
            "catboost_results.xlsx"
        ),

    "RandomForest":

        os.path.join(
            BENCHMARK_DIR,
            "RANDOM_FOREST_V3_FORECASTING",
            "rf_results.xlsx"
        )

}

# ==================================================
# NOWCASTING / FORECASTING FILES
# ==================================================

NOWCASTING_FILE = os.path.join(
    BENCHMARK_DIR,
    "LIGHTGBM",
    "lgbm_results.xlsx"
)

FORECASTING_FILE = os.path.join(
    BENCHMARK_DIR,
    "LIGHTGBM_V3_FORECASTING",
    "lgbm_results.xlsx"
)

# ==================================================
# LOAD FORECASTING RESULTS
# ==================================================

summary_rows = []

fold_results = {}

for model_name, file_path in FORECAST_MODELS.items():

    print(
        f"Loading {model_name}"
    )

    folds = pd.read_excel(
        file_path,
        sheet_name="FOLD_METRICS"
    )

    summary = pd.read_excel(
        file_path,
        sheet_name="SUMMARY"
    )

    fold_results[
        model_name
    ] = folds

    mae_mean = summary.loc[
        summary["Metric"] == "MAE",
        "Mean"
    ].values[0]

    mae_std = summary.loc[
        summary["Metric"] == "MAE",
        "Std"
    ].values[0]

    rmse_mean = summary.loc[
        summary["Metric"] == "RMSE",
        "Mean"
    ].values[0]

    rmse_std = summary.loc[
        summary["Metric"] == "RMSE",
        "Std"
    ].values[0]

    r2_mean = summary.loc[
        summary["Metric"] == "R2",
        "Mean"
    ].values[0]

    r2_std = summary.loc[
        summary["Metric"] == "R2",
        "Std"
    ].values[0]

    summary_rows.append([

        model_name,

        mae_mean,
        mae_std,

        rmse_mean,
        rmse_std,

        r2_mean,
        r2_std

    ])

# ==================================================
# FORECASTING SUMMARY
# ==================================================

forecast_df = pd.DataFrame(

    summary_rows,

    columns=[

        "Model",

        "MAE_Mean",
        "MAE_SD",

        "RMSE_Mean",
        "RMSE_SD",

        "R2_Mean",
        "R2_SD"

    ]

)

forecast_df = forecast_df.sort_values(
    "R2_Mean",
    ascending=False
)

forecast_df.to_csv(

    os.path.join(
        COMPARISON_DIR,
        "forecasting_models.csv"
    ),

    index=False

)

# ==================================================
# FORECASTING RANKING
# ==================================================

ranking_df = forecast_df.copy()

ranking_df["Rank"] = range(
    1,
    len(ranking_df)+1
)

ranking_df.to_csv(

    os.path.join(
        COMPARISON_DIR,
        "forecasting_ranking.csv"
    ),

    index=False

)

# ==================================================
# NOWCASTING vs FORECASTING
# ==================================================

now_summary = pd.read_excel(
    NOWCASTING_FILE,
    sheet_name="SUMMARY"
)

forecast_summary = pd.read_excel(
    FORECASTING_FILE,
    sheet_name="SUMMARY"
)

now_r2 = float(

    now_summary.loc[
        now_summary["Metric"] == "R2",
        "Mean"
    ].values[0]

)

forecast_r2 = float(

    forecast_summary.loc[
        forecast_summary["Metric"] == "R2",
        "Mean"
    ].values[0]

)

comparison_nf = pd.DataFrame({

    "Scenario": [

        "Nowcasting",
        "Forecasting"

    ],

    "Model": [

        "LightGBM",
        "LightGBM"

    ],

    "R2": [

        now_r2,
        forecast_r2

    ]

})

comparison_nf["Delta_vs_Nowcasting"] = [

    0,
    forecast_r2 - now_r2

]

comparison_nf.to_csv(

    os.path.join(
        COMPARISON_DIR,
        "nowcasting_vs_forecasting.csv"
    ),

    index=False

)

# ==================================================
# EXCEL SUMMARY
# ==================================================

with pd.ExcelWriter(

    os.path.join(
        COMPARISON_DIR,
        "benchmark_summary.xlsx"
    )

) as writer:

    forecast_df.to_excel(

        writer,

        sheet_name="FORECASTING",

        index=False

    )

    ranking_df.to_excel(

        writer,

        sheet_name="RANKING",

        index=False

    )

    comparison_nf.to_excel(

        writer,

        sheet_name="NOWCASTING_VS_FORECASTING",

        index=False

    )

# ==================================================
# FORECASTING R2 PLOT
# ==================================================

plt.figure(
    figsize=(8,5)
)

sns.barplot(

    data=forecast_df,

    x="Model",

    y="R2_Mean"

)

plt.title(
    "Forecasting Performance (R²)"
)

plt.tight_layout()

plt.savefig(

    os.path.join(
        COMPARISON_DIR,
        "forecasting_r2.png"
    ),

    dpi=300

)

plt.close()

# ==================================================
# FORECASTING RMSE PLOT
# ==================================================

plt.figure(
    figsize=(8,5)
)

sns.barplot(

    data=forecast_df,

    x="Model",

    y="RMSE_Mean"

)

plt.title(
    "Forecasting Performance (RMSE)"
)

plt.tight_layout()

plt.savefig(

    os.path.join(
        COMPARISON_DIR,
        "forecasting_rmse.png"
    ),

    dpi=300

)

plt.close()

# ==================================================
# NOWCASTING vs FORECASTING PLOT
# ==================================================

plt.figure(
    figsize=(6,4)
)

sns.barplot(

    data=comparison_nf,

    x="Scenario",

    y="R2"

)

plt.ylim(0,1)

plt.title(
    "LightGBM: Nowcasting vs Forecasting"
)

plt.tight_layout()

plt.savefig(

    os.path.join(
        COMPARISON_DIR,
        "nowcasting_vs_forecasting.png"
    ),

    dpi=300

)

plt.close()

# ==================================================
# FRIEDMAN TEST
# ==================================================

r2_matrix = []

for model_name in FORECAST_MODELS:

    r2_matrix.append(

        fold_results[
            model_name
        ]["R2"].values

    )

friedman_stat, friedman_p = (

    friedmanchisquare(
        *r2_matrix
    )

)

friedman_df = pd.DataFrame({

    "Statistic":
        [friedman_stat],

    "p_value":
        [friedman_p]

})

friedman_df.to_csv(

    os.path.join(
        COMPARISON_DIR,
        "friedman_test.csv"
    ),

    index=False

)

# ==================================================
# WILCOXON TESTS
# ==================================================

pairs = []

model_names = list(
    FORECAST_MODELS.keys()
)

for i in range(
    len(model_names)
):

    for j in range(
        i+1,
        len(model_names)
    ):

        m1 = model_names[i]

        m2 = model_names[j]

        stat, p = wilcoxon(

            fold_results[m1]["R2"],

            fold_results[m2]["R2"]

        )

        pairs.append([

            m1,
            m2,

            stat,
            p

        ])

pairwise_df = pd.DataFrame(

    pairs,

    columns=[

        "Model_1",
        "Model_2",

        "Statistic",
        "p_value"

    ]

)

pairwise_df.to_csv(

    os.path.join(
        COMPARISON_DIR,
        "pairwise_wilcoxon.csv"
    ),

    index=False

)

# ==================================================
# REPORT
# ==================================================

best_model = (
    forecast_df
    .iloc[0]["Model"]
)

best_r2 = (
    forecast_df
    .iloc[0]["R2_Mean"]
)

report = f"""
# BENCHMARK COMPARISON

## Forecasting Ranking

{ranking_df.to_string(index=False)}

## Best Forecasting Model

{best_model}

R2 = {best_r2:.4f}

## Nowcasting vs Forecasting

LightGBM Nowcasting:

R2 = {now_r2:.4f}

LightGBM Forecasting:

R2 = {forecast_r2:.4f}

Difference:

{forecast_r2 - now_r2:.4f}

## Friedman Test

Statistic:

{friedman_stat:.4f}

p-value:

{friedman_p:.6f}

## Interpretation

If p < 0.05:

Reject H0

At least one model behaves
significantly differently.

See pairwise_wilcoxon.csv
for pairwise comparisons.
"""

with open(

    os.path.join(
        COMPARISON_DIR,
        "BENCHMARK_REPORT.md"
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
print("COMPARISON COMPLETED")
print("===================================")

print(
    f"\nBest forecasting model: "
    f"{best_model}"
)

print(
    f"R² = {best_r2:.4f}"
)

print(
    f"\nNowcasting R² : "
    f"{now_r2:.4f}"
)

print(
    f"Forecasting R²: "
    f"{forecast_r2:.4f}"
)

print(
    f"\nFriedman p-value: "
    f"{friedman_p:.6f}"
)

print(
    f"\nResults:\n{COMPARISON_DIR}"
)

print(
    f"\nExecution time: "
    f"{elapsed:.2f} seconds"
)

print("\nDONE")