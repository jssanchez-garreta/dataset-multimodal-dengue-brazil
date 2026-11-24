"""
================================================================================
Script: 59_hae_ablation_and_weight_stability.py
Project: BraDengue-MM — Multimodal Dengue Surveillance & Forecasting in Brazil
================================================================================
Purpose:
    Isolates the contribution of horizon-aware simplex weighting from expert
    pool composition across 5 expanding-window folds (2021–2025) under strict
    boundary trimming (t <= T_train,end - h and t <= T_val,end - h). Compares:
      1. HAE (4-Expert: LightGBM + CatBoost + XGBoost + Naïve Persistence)
      2. Uniform 4-Expert Average (equal 0.25 weights across all 4 experts)
      3. 3-Expert GBDT-only HAE (simplex weights over GBDTs without Naïve)
      4. Uniform 3-GBDT Average (equal 1/3 weights over GBDTs)
    Also reports fold-to-fold variability (mean ± SD) of the learned simplex
    weights across horizons h = 1, 2, 4, 8 weeks.

Inputs:
    - FULL_FILE: Complete nationwide multimodal panel (4,634,540 records × 87 cols):
      `BraDengue-MM.parquet`
    - BENCH_FILE: Year-stratified 800,000-record benchmark subset:
      `dataset_benchmark_v1.parquet`

Outputs:
    - Console summary of out-of-sample MAE, RMSE, and R² (mean ± SD across folds)
      for all four ensemble variants and learned simplex weights (mean ± SD).
    - CSV file saved to `OUT_DIR/hae_ablation_weight_stability.csv`.
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
FULL_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL/BraDengue-MM.parquet"
BENCH_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/dataset_benchmark_v1.parquet"
OUT_DIR = "/content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/EXTENDED_EXPERIMENTS"
os.makedirs(OUT_DIR, exist_ok=True)
HORIZONS = [1, 2, 4, 8]

full_df = pd.read_parquet(FULL_FILE, columns=["codigo_ibge", "SE", "p_inc100k"]).sort_values(["codigo_ibge", "SE"]).reset_index(drop=True)
full_df["persist_base"] = full_df["p_inc100k"].astype(np.float32)
for h in HORIZONS:
    full_df[f"target_h{h}"] = full_df.groupby("codigo_ibge")["p_inc100k"].shift(-h).astype(np.float32)
target_cols = ["persist_base"] + [f"target_h{h}" for h in HORIZONS]

bench = pd.read_parquet(BENCH_FILE).merge(full_df[["codigo_ibge", "SE"] + target_cols], on=["codigo_ibge", "SE"], how="left")
del full_df

se_sorted = np.sort(bench["SE"].unique())
se_to_idx = {se: i for i, se in enumerate(se_sorted)}
bench["t_idx"] = bench["SE"].map(se_to_idx).astype(np.int32)
year_max_t = bench.groupby("anio")["t_idx"].max().to_dict()
bench = bench.dropna(subset=["target_h8"]).reset_index(drop=True)

LEAKAGE = {
    "Unnamed: 0", "codigo_ibge", "municipio", "estado", "SE", "anio", "semana", "data_iniSE",
    "casos_est", "casos_est_min", "casos_est_max", "nivel", "nivel_inc",
    "casprov", "casos", "Rt", "p_rt1", "transmissao", "receptivo", "notif_accum_year",
    "dominant_land_cover", "persist_base", "t_idx"
} | {f"target_h{h}" for h in HORIZONS}
features = [c for c in bench.columns if c not in LEAKAGE and pd.api.types.is_numeric_dtype(bench[c])]


def fit_simplex_weights(P, y):
    K = P.shape[-1]
    res = minimize(
        lambda w: np.mean((y - P @ w) ** 2),
        np.ones(K) / K,
        bounds=[(0.0, 1.0)] * K,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
        options={"ftol": 1e-9, "maxiter": 500}
    )
    return res.x if res.success else np.ones(K) / K


folds = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
names = ["HAE_4Expert", "Uniform_4Expert", "HAE_3GBDT", "Uniform_3GBDT"]
res_store = {(h, m): [] for h in HORIZONS for m in names}
w4_folds = {h: [] for h in HORIZONS}

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

        imp_v = SimpleImputer(strategy="median")
        X_sub = imp_v.fit_transform(bench.loc[sub_mask, features].astype(np.float32))
        X_val = imp_v.transform(bench.loc[val_mask, features].astype(np.float32))
        y_sub = bench.loc[sub_mask, tcol].values.astype(np.float32)
        y_val = bench.loc[val_mask, tcol].values.astype(np.float32)
        p_nai_val = bench.loc[val_mask, "persist_base"].values.astype(np.float32)

        p_lgb_v = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=127, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1).fit(X_sub, y_sub).predict(X_val)
        p_cat_v = CatBoostRegressor(iterations=80, learning_rate=0.08, depth=4, rsm=0.8, border_count=64, random_seed=SEED, thread_count=-1, verbose=0).fit(X_sub, y_sub).predict(X_val)
        p_xgb_v = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED).fit(X_sub, y_sub).predict(X_val)

        w4 = fit_simplex_weights(np.column_stack([p_lgb_v, p_cat_v, p_xgb_v, p_nai_val]), y_val)
        w3 = fit_simplex_weights(np.column_stack([p_lgb_v, p_cat_v, p_xgb_v]), y_val)
        w4_folds[h].append(w4)

        imp_f = SimpleImputer(strategy="median")
        X_tr = imp_f.fit_transform(bench.loc[tr_mask, features].astype(np.float32))
        X_te = imp_f.transform(X_te_raw)
        y_tr = bench.loc[tr_mask, tcol].values.astype(np.float32)
        y_te = bench.loc[te_mask, tcol].values.astype(np.float32)

        p_lgb = LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=15, max_bin=127, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1).fit(X_tr, y_tr).predict(X_te)
        p_cat = CatBoostRegressor(iterations=80, learning_rate=0.08, depth=4, rsm=0.8, border_count=64, random_seed=SEED, thread_count=-1, verbose=0).fit(X_tr, y_tr).predict(X_te)
        p_xgb = XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=SEED).fit(X_tr, y_tr).predict(X_te)

        preds = {
            "HAE_4Expert": np.column_stack([p_lgb, p_cat, p_xgb, p_nai_te]) @ w4,
            "Uniform_4Expert": 0.25 * (p_lgb + p_cat + p_xgb + p_nai_te),
            "HAE_3GBDT": np.column_stack([p_lgb, p_cat, p_xgb]) @ w3,
            "Uniform_3GBDT": (p_lgb + p_cat + p_xgb) / 3.0,
        }
        for m, pr in preds.items():
            res_store[(h, m)].append((mean_absolute_error(y_te, pr), np.sqrt(mean_squared_error(y_te, pr)), r2_score(y_te, pr)))
    print(f"Fold {fold_id}/5 done in {(time.time()-t0):.1f}s", flush=True)

rows = []
for h in HORIZONS:
    print(f"\n=== h = {h} ===")
    for m in names:
        arr = np.array(res_store[(h, m)])
        print(f"{m:18s} | MAE={arr[:,0].mean():.2f}±{arr[:,0].std():.2f} | RMSE={arr[:,1].mean():.2f}±{arr[:,1].std():.2f} | R2={arr[:,2].mean():.4f}±{arr[:,2].std():.4f}")
        rows.append({
            "Horizon_Weeks": h, "Model": m,
            "MAE_Mean": arr[:,0].mean(), "MAE_SD": arr[:,0].std(),
            "RMSE_Mean": arr[:,1].mean(), "RMSE_SD": arr[:,1].std(),
            "R2_Mean": arr[:,2].mean(), "R2_SD": arr[:,2].std(),
        })
    W = np.array(w4_folds[h])
    print(f"Weights (Mean±SD): LGBM={W[:,0].mean():.3f}±{W[:,0].std():.3f}, Cat={W[:,1].mean():.3f}±{W[:,1].std():.3f}, XGB={W[:,2].mean():.3f}±{W[:,2].std():.3f}, Naive={W[:,3].mean():.3f}±{W[:,3].std():.3f}")

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "hae_ablation_weight_stability.csv"), index=False)