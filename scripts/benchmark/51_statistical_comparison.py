# ============================================================
# 51_statistical_comparison.py
#
# Statistical Comparison Framework
#
# ============================================================

import os

import numpy as np
import pandas as pd

from itertools import combinations

from scipy.stats import (
    wilcoxon
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    f1_score,
    precision_score,
    recall_score
)

# ==================================================
# CONFIG
# ==================================================

PROJECT_DIR = (
    "/content/drive/MyDrive/DENGUE_BRASIL"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "BENCHMARK",
    "STATISTICAL_COMPARISON"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

COMPARISON_XLSX = os.path.join(
    OUTPUT_DIR,
    "statistical_comparison.xlsx"
)

REPORT_MD = os.path.join(
    OUTPUT_DIR,
    "STATISTICAL_REPORT.md"
)

# ==================================================
# MODEL FILES
# ==================================================

MODELS = {

    "RF":
    os.path.join(
        PROJECT_DIR,
        "BENCHMARK",
        "RANDOM_FOREST_V3_FORECASTING",
        "rf_predictions.csv"
    ),

    "XGB":
    os.path.join(
        PROJECT_DIR,
        "BENCHMARK",
        "XGBOOST_V3_FORECASTING",
        "xgb_predictions.csv"
    ),

    "CATBOOST":
    os.path.join(
        PROJECT_DIR,
        "BENCHMARK",
        "CATBOOST_V3_FORECASTING",
        "catboost_predictions.csv"
    ),

    "LGBM":
    os.path.join(
        PROJECT_DIR,
        "BENCHMARK",
        "LIGHTGBM_V3_FORECASTING",
        "lgbm_predictions.csv"
    ),

    "MLP":
    os.path.join(
        PROJECT_DIR,
        "BENCHMARK",
        "MLP_V2_FORECASTING",
        "mlp_predictions.csv"
    ),

    "EPI_MLP":
    os.path.join(
        PROJECT_DIR,
        "BENCHMARK",
        "EPI_MLP_V1_FORECASTING",
        "epi_mlp_predictions.csv"
    ),

    "EPI_MLP_V2":
    os.path.join(
        PROJECT_DIR,
        "BENCHMARK",
        "EPI_MLP_V2_FORECASTING",
        "epi_mlp_v2_predictions.csv"
    ),

    "EPI_MLP_V3":
    os.path.join(
        PROJECT_DIR,
        "BENCHMARK",
        "EPI_MLP_V3_FORECASTING",
        "epi_mlp_v3_predictions.csv"
    ),

    "EPI_MLP_SIR":
    os.path.join(
        PROJECT_DIR,
        "BENCHMARK",
        "EPI_MLP_SIR_V1_FORECASTING",
        "epi_mlp_sir_predictions.csv"
    ),

    "EPI_MLP_GROWTH_MT":
    os.path.join(
        PROJECT_DIR,
        "BENCHMARK",
        "EPI_MLP_GROWTH_MULTITASK_V1_FORECASTING",
        "epi_mlp_growth_multitask_predictions.csv"
    ),

    "EPI_MLP_OUTBREAK_MT":
    os.path.join(
        PROJECT_DIR,
        "BENCHMARK",
        "EPI_MLP_OUTBREAK_MULTITASK_V1_FORECASTING",
        "epi_mlp_outbreak_multitask_predictions.csv"
    )
}

# ==================================================
# LOAD PREDICTIONS
# ==================================================

dfs = {}

for name, path in MODELS.items():

    if os.path.exists(path):

        df = pd.read_csv(path)

        dfs[name] = df

        print(
            f"Loaded {name}: "
            f"{len(df):,} rows"
        )

        print(
            f"Years: "
            f"{df['anio'].min()}-"
            f"{df['anio'].max()}"
        )

print("\nLoaded models:")
print(list(dfs.keys()))

print("\nNumber of loaded models:")
print(len(dfs))

# ==================================================
# CLASSICAL METRICS
# ==================================================

summary_rows = []

for name, df in dfs.items():

    real = df["real"].values

    pred = df["pred"].values

    mae = mean_absolute_error(
        real,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            real,
            pred
        )
    )

    r2 = r2_score(
        real,
        pred
    )

    summary_rows.append([
        name,
        mae,
        rmse,
        r2,
        len(df)
    ])

summary_df = pd.DataFrame(

    summary_rows,

    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2",
        "Samples"
    ]
)

print("\nSummary DF shape:")
print(summary_df.shape)

print(summary_df.head())

# ==================================================
# DELTA VS MLP AND BEST MODEL
# ==================================================
print(summary_df["Model"].tolist())

if len(summary_df) == 0:

    raise ValueError(
        "summary_df is empty. "
        "No models were loaded correctly."
    )

if "MLP" not in summary_df["Model"].values:

    raise ValueError(
        f"MLP not found. "
        f"Available models: "
        f"{summary_df['Model'].tolist()}"
    )

mlp_r2 = float(
    summary_df.loc[
        summary_df["Model"] == "MLP",
        "R2"
    ].iloc[0]
)

summary_df["R2_Delta_vs_MLP"] = (
    summary_df["R2"]
    -
    mlp_r2
)

best_r2 = summary_df["R2"].max()

best_model = (
    summary_df.loc[
        summary_df["R2"].idxmax(),
        "Model"
    ]
)

summary_df["R2_Delta_vs_BestModel"] = (
    summary_df["R2"]
    -
    best_r2
)

# ==================================================
# WILCOXON TESTS
# ==================================================

from itertools import combinations

pairs = list(
    combinations(
        dfs.keys(),
        2
    )
)

wilcoxon_rows = []

for a, b in pairs:

    merged = (

        dfs[a]

        .merge(

            dfs[b],

            on=[
                "codigo_ibge",
                "anio",
                "semana"
            ],

            suffixes=(
                "_a",
                "_b"
            )
        )
    )

    if len(merged) == 0:
        continue

    # -----------------------------------------
    # GLOBAL
    # -----------------------------------------

    err_a = np.abs(
        merged["real_a"]
        -
        merged["pred_a"]
    )

    err_b = np.abs(
        merged["real_b"]
        -
        merged["pred_b"]
    )

    # -----------------------------------------
    # WINNER
    # -----------------------------------------

    if err_a.mean() < err_b.mean():

        winner = a

    else:
        winner = b

    stat_global, p_global = wilcoxon(
        err_a,
        err_b,
        zero_method="zsplit"
    )

    # -----------------------------------------
    # P90 OUTBREAKS
    # -----------------------------------------

    p90 = np.percentile(
        merged["real_a"],
        90
    )

    peak_mask = (
        merged["real_a"]
        >=
        p90
    )

    stat_p90 = np.nan
    pvalue_p90 = np.nan

    if peak_mask.sum() > 10:

        stat_p90, pvalue_p90 = wilcoxon(
            err_a[peak_mask],
            err_b[peak_mask],
            zero_method="zsplit"
        )

    # -----------------------------------------
    # P95 EXTREME OUTBREAKS
    # -----------------------------------------

    p95 = np.percentile(
        merged["real_a"],
        95
    )

    extreme_mask = (
        merged["real_a"]
        >=
        p95
    )

    stat_p95 = np.nan
    pvalue_p95 = np.nan

    if extreme_mask.sum() > 10:

        stat_p95, pvalue_p95 = wilcoxon(
            err_a[extreme_mask],
            err_b[extreme_mask],
            zero_method="zsplit"
        )

    wilcoxon_rows.append([

        a,
        b,

        winner,

        err_a.mean(),
        err_b.mean(),

        stat_global,
        p_global,

        stat_p90,
        pvalue_p90,

        stat_p95,
        pvalue_p95
    ])

wilcoxon_df = pd.DataFrame(

    wilcoxon_rows,

    columns=[

        "Model_A",
        "Model_B",

        "Winner",

        "Mean_Error_A",
        "Mean_Error_B",

        "Global_Statistic",
        "Global_PValue",

        "P90_Statistic",
        "P90_PValue",

        "P95_Statistic",
        "P95_PValue"
    ]
)

# ==================================================
# SIGNIFICANCE FLAGS
# ==================================================

wilcoxon_df["Global_Significant"] = (
    wilcoxon_df["Global_PValue"] < 0.05
)

wilcoxon_df["P90_Significant"] = (
    wilcoxon_df["P90_PValue"] < 0.05
)

wilcoxon_df["P95_Significant"] = (
    wilcoxon_df["P95_PValue"] < 0.05
)

wilcoxon_df["Significant_Any"] = (
    wilcoxon_df["Global_Significant"]
    |
    wilcoxon_df["P90_Significant"]
    |
    wilcoxon_df["P95_Significant"]
)

# ==================================================
# SIGNIFICANCE PERCENTAGES
# ==================================================

n_tests = len(wilcoxon_df)

global_pct = (
    100 *
    wilcoxon_df["Global_Significant"].sum()
    /
    max(n_tests, 1)
)

p90_pct = (
    100 *
    wilcoxon_df["P90_Significant"].sum()
    /
    max(n_tests, 1)
)

p95_pct = (
    100 *
    wilcoxon_df["P95_Significant"].sum()
    /
    max(n_tests, 1)
)

# ==================================================
# SAFE PERCENTILE
# ==================================================

def safe_percentile(values, q):

    if len(values) == 0:
        return np.nan

    return np.percentile(
        values,
        q
    )

# ==================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ==================================================

boot_rows = []

N_BOOT = 500

for name, df in dfs.items():

    real = df["real"].values
    pred = df["pred"].values

    n = len(df)

    mae_scores = []
    rmse_scores = []
    r2_scores = []

    peak_rmse_scores = []
    peak_bias_pct_scores = []
    peak_f1_scores = []
    peak_precision_scores = []
    peak_recall_scores = []

    extreme_f1_scores = []
    extreme_precision_scores = []
    extreme_recall_scores = []

    municipality_median_r2_scores = []

    for _ in range(N_BOOT):

        idx = np.random.randint(
            0,
            n,
            n
        )

        real_boot = real[idx]
        pred_boot = pred[idx]

        # --------------------------------------
        # CLASSICAL METRICS
        # --------------------------------------

        mae_scores.append(
            mean_absolute_error(
                real_boot,
                pred_boot
            )
        )

        rmse_scores.append(
            np.sqrt(
                mean_squared_error(
                    real_boot,
                    pred_boot
                )
            )
        )

        r2_scores.append(
            r2_score(
                real_boot,
                pred_boot
            )
        )

        # --------------------------------------
        # P90 OUTBREAKS
        # --------------------------------------

        p90 = safe_percentile(
            real_boot,
            90
        )

        peak_mask = (
            real_boot >= p90
        )

        if peak_mask.sum() > 10:

            peak_real = real_boot[
                peak_mask
            ]

            peak_pred = pred_boot[
                peak_mask
            ]

            peak_rmse_scores.append(

                np.sqrt(
                    mean_squared_error(
                        peak_real,
                        peak_pred
                    )
                )
            )

            peak_bias = (

                peak_pred.mean()

                -

                peak_real.mean()
            )

            peak_bias_pct_scores.append(

                100.0

                *

                peak_bias

                /

                (
                    peak_real.mean()
                    + 1e-6
                )
            )

            real_peak_class = (
                real_boot >= p90
            ).astype(int)

            pred_peak_class = (
                pred_boot >= p90
            ).astype(int)

            peak_precision_scores.append(

                precision_score(

                    real_peak_class,

                    pred_peak_class,

                    zero_division=0
                )
            )

            peak_recall_scores.append(

                recall_score(

                    real_peak_class,

                    pred_peak_class,

                    zero_division=0
                )
            )

            peak_f1_scores.append(

                f1_score(

                    real_peak_class,

                    pred_peak_class,

                    zero_division=0
                )
            )

        # --------------------------------------
        # P95 EXTREME OUTBREAKS
        # --------------------------------------

        p95 = safe_percentile(
            real_boot,
            95
        )

        real_extreme = (
            real_boot >= p95
        ).astype(int)

        pred_extreme = (
            pred_boot >= p95
        ).astype(int)

        extreme_precision_scores.append(

            precision_score(

                real_extreme,

                pred_extreme,

                zero_division=0
            )
        )

        extreme_recall_scores.append(

            recall_score(

                real_extreme,

                pred_extreme,

                zero_division=0
            )
        )

        extreme_f1_scores.append(

            f1_score(

                real_extreme,

                pred_extreme,

                zero_division=0
            )
        )

    # ------------------------------------------
    # SAVE CONFIDENCE INTERVALS
    # ------------------------------------------

    boot_rows.append([

        name,

        safe_percentile(mae_scores, 2.5),
        safe_percentile(mae_scores,97.5),

        safe_percentile(rmse_scores, 2.5),
        safe_percentile(rmse_scores,97.5),

        safe_percentile(r2_scores, 2.5),
        safe_percentile(r2_scores,97.5),

        safe_percentile(
            peak_rmse_scores,
            2.5
        ),
        safe_percentile(
            peak_rmse_scores,
            97.5
        ),

        safe_percentile(
            peak_bias_pct_scores,
            2.5
        ),
        safe_percentile(
            peak_bias_pct_scores,
            97.5
        ),

        safe_percentile(
            peak_f1_scores,
            2.5
        ),
        safe_percentile(
            peak_f1_scores,
            97.5
        ),

        safe_percentile(
            extreme_f1_scores,
            2.5
        ),
        safe_percentile(
            extreme_f1_scores,
            97.5
        ),
        safe_percentile(
            peak_precision_scores,
            2.5
        ),
        safe_percentile(
            peak_precision_scores,
            97.5
        ),

        safe_percentile(
            peak_recall_scores,
            2.5
        ),
        safe_percentile(
            peak_recall_scores,
            97.5
        ),

        safe_percentile(
            extreme_precision_scores,
            2.5
        ),
        safe_percentile(
            extreme_precision_scores,
            97.5
        ),

        safe_percentile(
            extreme_recall_scores,
            2.5
        ),
        safe_percentile(
            extreme_recall_scores,
            97.5
        )
    ])

bootstrap_df = pd.DataFrame(

    boot_rows,

    columns=[

        "Model",

        "MAE_CI_Lower",
        "MAE_CI_Upper",

        "RMSE_CI_Lower",
        "RMSE_CI_Upper",

        "R2_CI_Lower",
        "R2_CI_Upper",

        "Peak_RMSE_CI_Lower",
        "Peak_RMSE_CI_Upper",

        "Peak_BiasPct_CI_Lower",
        "Peak_BiasPct_CI_Upper",

        "Peak_F1_CI_Lower",
        "Peak_F1_CI_Upper",

        "Extreme_F1_CI_Lower",
        "Extreme_F1_CI_Upper",

        "Peak_Precision_CI_Lower",
        "Peak_Precision_CI_Upper",

        "Peak_Recall_CI_Lower",
        "Peak_Recall_CI_Upper",

        "Extreme_Precision_CI_Lower",
        "Extreme_Precision_CI_Upper",

        "Extreme_Recall_CI_Lower",
        "Extreme_Recall_CI_Upper"
    ]
)

# ==================================================
# SAVE
# ==================================================

with pd.ExcelWriter(
    COMPARISON_XLSX
) as writer:

    summary_sorted = (
        summary_df.sort_values(
            "R2",
            ascending=False
        ).reset_index(
            drop=True
        )
    )

    summary_sorted.insert(
        0,
        "Rank",
        range(
            1,
            len(summary_sorted) + 1
        )
    )

    mlp_r2 = float(
        summary_df.loc[
            summary_df["Model"] == "MLP",
            "R2"
        ].iloc[0]
    )

    summary_sorted["Delta_R2_vs_MLP"] = (
        summary_sorted["R2"] - mlp_r2
    )
    
    best_r2 = summary_sorted["R2"].max()

    summary_df["R2_Delta_vs_BestModel"] = (
        summary_sorted["R2"] - best_r2
    )

    summary_sorted = summary_sorted.round(4)

    summary_sorted.to_excel(
        writer,
        sheet_name="METRICS",
        index=False
    )

    wilcoxon_df.to_excel(
        writer,
        sheet_name="WILCOXON",
        index=False
    )

    bootstrap_df.to_excel(
        writer,
        sheet_name="BOOTSTRAP_CI",
        index=False
    )

# ==================================================
# REPORT
# ==================================================

best_model = summary_sorted.iloc[0]["Model"]

best_r2 = summary_sorted.iloc[0]["R2"]

best_delta_vs_mlp = float(

    summary_sorted.loc[
        summary_sorted["Model"] == best_model,
        "Delta_R2_vs_MLP"
    ].iloc[0]
)

report = f"""
# Statistical Comparison

Models compared:
{len(dfs)}

Best Model:
{best_model}

Best R2:
{best_r2:.4f}

Mean Delta vs MLP:
{best_delta_vs_mlp:.4f}

Total Pairwise Comparisons:
{n_tests}

Wilcoxon Tests:
{len(wilcoxon_df)}

Bootstrap Iterations:
{N_BOOT}

Significant Comparisons:

Global:
{wilcoxon_df["Global_Significant"].sum()}
({global_pct:.1f}%)

P90:
{wilcoxon_df["P90_Significant"].sum()}
({p90_pct:.1f}%)

P95:
{wilcoxon_df["P95_Significant"].sum()}
({p95_pct:.1f}%)
"""

with open(
    REPORT_MD,
    "w",
    encoding="utf-8"
) as f:
    f.write(report)

print(summary_sorted)

print(
    "\nSignificant Comparisons"
)

print(
    f"Global : "
    f"{wilcoxon_df['Global_Significant'].sum()} "
    f"({global_pct:.1f}%)"
)

print(
    f"P90    : "
    f"{wilcoxon_df['P90_Significant'].sum()} "
    f"({p90_pct:.1f}%)"
)

print(
    f"P95    : "
    f"{wilcoxon_df['P95_Significant'].sum()} "
    f"({p95_pct:.1f}%)"
)

print(
    f"\nBest model: {best_model}"
)

print(
    f"Best R2: {best_r2:.4f}"
)

print(
    f"Best Model Advantage vs MLP: "
    f"{best_delta_vs_mlp:.4f}"
)

print("\nDONE")