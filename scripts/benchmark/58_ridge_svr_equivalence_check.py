"""
================================================================================
Script: 58_ridge_svr_equivalence_check.py
Project: BraDengue-MM — Multimodal Dengue Surveillance & Forecasting in Brazil
================================================================================
Purpose:
    Reproducibility sanity check verifying independent training and numerical
    equivalence between Ridge Regression (alpha = 1.0) and primal LinearSVR
    (C = 1.0, epsilon = 0, squared epsilon-insensitive loss) across all 5
    expanding-window test folds (2021–2025) at h = 1 week. Reports 6-decimal
    R² values, Pearson correlation between out-of-sample predictions, and
    maximum absolute prediction difference per fold.

Inputs:
    - FULL_FILE: Complete nationwide multimodal panel (4,634,540 records × 87 cols)
      used to compute causal 1-week-ahead targets (`target_h1`):
      `BraDengue-MM.parquet`
    - BENCH_FILE: Year-stratified 800,000-record benchmark subset:
      `dataset_benchmark_v1.parquet`

Outputs:
    - Console diagnostic summary per fold (2021–2025) reporting 6-decimal R²,
      prediction Pearson correlation (`Corr`), and maximum absolute difference
      (`MaxDiff`).
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FULL_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL/BraDengue-MM.parquet"
BENCH_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/dataset_benchmark_v1.parquet"

full_df = pd.read_parquet(FULL_FILE, columns=["codigo_ibge", "SE", "p_inc100k"]).sort_values(["codigo_ibge", "SE"]).reset_index(drop=True)
full_df["target_h1"] = full_df.groupby("codigo_ibge")["p_inc100k"].shift(-1).astype(np.float32)
bench = pd.read_parquet(BENCH_FILE).merge(full_df[["codigo_ibge", "SE", "target_h1"]], on=["codigo_ibge", "SE"], how="left").dropna(subset=["target_h1"])

LEAKAGE = {"Unnamed: 0", "codigo_ibge", "municipio", "estado", "SE", "anio", "semana", "data_iniSE", "casos_est", "casos_est_min", "casos_est_max", "nivel", "nivel_inc", "casprov", "casos", "Rt", "p_rt1", "transmissao", "receptivo", "notif_accum_year", "dominant_land_cover", "target_h1"}
features = [c for c in bench.columns if c not in LEAKAGE and pd.api.types.is_numeric_dtype(bench[c])]

for tr_end, te_yr in [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]:
    tr = bench["anio"] <= tr_end
    te = bench["anio"] == te_yr
    imp = SimpleImputer(strategy="median")
    sc = StandardScaler()
    X_tr = sc.fit_transform(imp.fit_transform(bench.loc[tr, features].astype(np.float32)))
    X_te = sc.transform(imp.transform(bench.loc[te, features].astype(np.float32)))
    y_tr = bench.loc[tr, "target_h1"].values
    y_te = bench.loc[te, "target_h1"].values

    p_rid = Ridge(alpha=1.0).fit(X_tr, y_tr).predict(X_te)
    p_svr = LinearSVR(C=1.0, dual=False, loss="squared_epsilon_insensitive", random_state=42).fit(X_tr, y_tr).predict(X_te)
    corr = np.corrcoef(p_rid, p_svr)[0, 1]
    max_diff = np.max(np.abs(p_rid - p_svr))
    print(f"Fold {te_yr}: Ridge R2={r2_score(y_te, p_rid):.6f} | LinearSVR R2={r2_score(y_te, p_svr):.6f} | Corr={corr:.6f} | MaxDiff={max_diff:.5f}")
