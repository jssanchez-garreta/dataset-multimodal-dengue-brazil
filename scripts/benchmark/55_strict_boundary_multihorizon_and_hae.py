"""
================================================================================
Script: 55_strict_boundary_multihorizon_and_hae.py
Project: BraDengue-MM — Multimodal Dengue Surveillance & Forecasting in Brazil
================================================================================
Purpose:
    Evaluates multi-horizon prospective dengue forecasting (h = 1, 2, 4, 8 weeks)
    across 5 expanding-window temporal folds (2021–2025) under strict boundary
    trimming (t <= T_train,end - h and t <= T_val,end - h) to prevent cross-year
    label spillover. Benchmarks LightGBM, CatBoost, XGBoost, Naïve Persistence,
    uniform GBDT averaging, and the simplex-constrained Horizon-Adaptive
    Ensemble (HAE) solved via SLSQP.

Inputs:
    - FULL_FILE: Complete nationwide multimodal panel (4,634,540 records × 87 cols)
      used to compute leak-free future targets (target_h1..target_h8) and
      persistence baselines prior to subset alignment:
      `BraDengue-MM.parquet`
    - BENCH_FILE: Year-stratified 800,000-record benchmark subset:
      `dataset_benchmark_v1.parquet`

Outputs:
    - Console summary of out-of-sample MAE, RMSE, and R² (mean ± SD across folds)
      for all models and learned HAE simplex weights across lead times.
    - CSV file saved to `OUT_DIR/strict_boundary_multihorizon_hae.csv`.
================================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

FULL_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL/BraDengue-MM.parquet"
BENCH_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/dataset_benchmark_v1.parquet"
OUT_DIR = "/content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/EXTENDED_EXPERIMENTS"
os.makedirs(OUT_DIR, exist_ok=True)

HORIZONS = [1, 2, 4, 8]

# 1. Compute causal horizon-shifted targets on the continuous 4.63M panel first
full_df = (
    pd.read_parquet(FULL_FILE, columns=["codigo_ibge", "SE", "p_inc100k"])
    .sort_values(["codigo_ibge", "SE"])
    .reset_index(drop=True)
)
full_df["persist_base"] = full_df["p_inc100k"].astype(np.float32)
for h in HORIZONS:
    full_df[f"target_h{h}"] = full_df.groupby("codigo_ibge")["p_inc100k"].shift(-h).astype(np.float32)
target_cols = ["persist_base"] + [f"target_h{h}" for h in HORIZONS]

bench = pd.read_parquet(BENCH_FILE).merge(
    full_df[["codigo_ibge", "SE"] + target_cols],
    on=["codigo_ibge", "SE"],
    how="left",
)
del full_df

# Global chronological week index (0..831) for strict boundary trimming
se_sorted = np.sort(bench["SE"].unique())
se_to_idx = {se: i for i, se in enumerate(se_sorted)}
bench["t_idx"] = bench["SE"].map(se_to_idx).astype(np.int32)
year_max_t = bench.groupby("anio")["t_idx"].max().to_dict()
bench = bench.dropna(subset=[f"target_h{max(HORIZONS)}"]).reset_index(drop=True)

LEAKAGE = {
    "Unnamed: 0", "codigo_ibge", "municipio", "estado", "SE", "anio", "semana", "data_iniSE",
    "casos_est", "casos_est_min", "casos_est_max", "nivel", "nivel_inc",
    "casprov", "casos", "Rt", "p_rt1", "transmissao", "receptivo", "notif_accum_year",
    "dominant_land_cover", "persist_base", "t_idx",
} | {f"target_h{h}" for h in HORIZONS}
features = [c for c in bench.columns if c not in LEAKAGE and pd.api.types.is_numeric_dtype(bench[c])]
print(f"Causal predictors count: {len(features)}")


def fit_simplex_weights(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    K = P.shape[-1]
    res = minimize(
        lambda w: np.mean((y - P @ w) ** 2),
        np.ones(K) / K,
        bounds=[(0.0, 1.0)] * K,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
        options={"ftol": 1e-9, "maxiter": 500},
    )
    return res.x if res.success else np.ones(K) / K


folds = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
models_fast = ["HorizonAdaptiveEnsemble", "EqualGBDT_Ensemble", "LightGBM", "CatBoost", "XGBoost", "NaivePersistence"]
res_store = {(h, m): [] for h in HORIZONS for m in models_fast}
w_folds = {h: [] for h in HORIZONS}

xgb_kw = dict(tree_method="hist", n_jobs=-1)
cat_kw = dict(thread_count=-1, iterations=100, depth=4, rsm=0.8, border_count=64)

t0 = time.time()
for fold_id, (tr_end, te_yr) in enumerate(folds, 1):
    T_tr_end = year_max_t[tr_end]
    T_sub_end = year_max_t[tr_end - 1]
    te_mask = bench["anio"] == te_yr
    X_te_raw = bench.loc[te_mask, features].astype(np.float32)
    p_nai_te = bench.loc[te_mask, "persist_base"].values.astype(np.float32)

    for h in HORIZONS:
        tcol = f"target_h{h}"
        sub_mask = bench["t_idx"] <= (T_sub_end - h)
        val_mask = (bench["anio"] == tr_end) & (bench["t_idx"] <= (T_tr_end - h))
        tr_mask = bench["t_idx"] <= (T_tr_end - h)

        # Internal pre-test validation to fit simplex weights
        imp_v = SimpleImputer(strategy="median")
        X_sub = imp_v.fit_transform(bench.loc[sub_mask, features].astype(np.float32))
        X_val = imp_v.transform(bench.loc[val_mask, features].astype(np.float32))
        y_sub = bench.loc[sub_mask, tcol].values.astype(np.float32)
        y_val = bench.loc[val_mask, tcol].values.astype(np.float32)
        p_nai_val = bench.loc[val_mask, "persist_base"].values.astype(np.float32)

        p_lgb_v = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, max_bin=127, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1).fit(X_sub, y_sub).predict(X_val)
        p_cat_v = CatBoostRegressor(learning_rate=0.05, random_seed=SEED, verbose=0, **cat_kw).fit(X_sub, y_sub).predict(X_val)
        p_xgb_v = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=SEED, **xgb_kw).fit(X_sub, y_sub).predict(X_val)
        w_opt = fit_simplex_weights(np.column_stack([p_lgb_v, p_cat_v, p_xgb_v, p_nai_val]), y_val)
        w_folds[h].append(w_opt)

        # Full boundary-trimmed fold training and out-of-sample test evaluation
        imp_f = SimpleImputer(strategy="median")
        X_tr = imp_f.fit_transform(bench.loc[tr_mask, features].astype(np.float32))
        X_te = imp_f.transform(X_te_raw)
        y_tr = bench.loc[tr_mask, tcol].values.astype(np.float32)
        y_te = bench.loc[te_mask, tcol].values.astype(np.float32)

        p_lgb = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, max_bin=127, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1).fit(X_tr, y_tr).predict(X_te)
        p_cat = CatBoostRegressor(learning_rate=0.05, random_seed=SEED, verbose=0, **cat_kw).fit(X_tr, y_tr).predict(X_te)
        p_xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=SEED, **xgb_kw).fit(X_tr, y_tr).predict(X_te)
        p_hae = np.column_stack([p_lgb, p_cat, p_xgb, p_nai_te]) @ w_opt
        p_eq = (p_lgb + p_cat + p_xgb) / 3.0

        for name, pr in [
            ("HorizonAdaptiveEnsemble", p_hae),
            ("EqualGBDT_Ensemble", p_eq),
            ("LightGBM", p_lgb),
            ("CatBoost", p_cat),
            ("XGBoost", p_xgb),
            ("NaivePersistence", p_nai_te),
        ]:
            res_store[(h, name)].append((
                mean_absolute_error(y_te, pr),
                np.sqrt(mean_squared_error(y_te, pr)),
                r2_score(y_te, pr),
            ))

    print(f"Fold {fold_id}/5 done in {(time.time() - t0):.1f}s", flush=True)

rows = []
for h in HORIZONS:
    print(f"\n--- h = {h} ---")
    for m in models_fast:
        arr = np.array(res_store[(h, m)])
        print(f"{m:25s} | MAE={arr[:,0].mean():.2f}±{arr[:,0].std():.2f} | RMSE={arr[:,1].mean():.2f}±{arr[:,1].std():.2f} | R2={arr[:,2].mean():.4f}±{arr[:,2].std():.4f}")
        rows.append({
            "Horizon_Weeks": h, "Model": m,
            "MAE_Mean": arr[:,0].mean(), "MAE_SD": arr[:,0].std(),
            "RMSE_Mean": arr[:,1].mean(), "RMSE_SD": arr[:,1].std(),
            "R2_Mean": arr[:,2].mean(), "R2_SD": arr[:,2].std(),
        })
    wm = np.mean(w_folds[h], axis=0)
    print(f"Weights: LGBM={wm.item(0):.3f}, Cat={wm.item(1):.3f}, XGB={wm.item(2):.3f}, Naive={wm.item(3):.3f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "strict_boundary_multihorizon_hae.csv"), index=False)
