# ============================================================
# 53_horizon_adaptive_ensemble.py
#
# PROJECT:
# Early Detection of Dengue Outbreaks in Brazil
#
# OBJECTIVE:
# Evaluate a leakage-free Horizon-Adaptive Ensemble (HAE) that
# dynamically learns non-negative simplex combination weights
# across complementary GBDT paradigms (LightGBM, CatBoost, XGBoost)
# and Naive Persistence via internal temporal validation at each
# forecasting horizon h in {1, 2, 4, 8} weeks.
#
# INPUTS:
# - /content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/dataset_benchmark_v1.parquet
# - /content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL/BraDengue-MM.parquet
#
# OUTPUTS (saved in DENGUE_BRASIL/BENCHMARK/EXTENDED_EXPERIMENTS/):
# - horizon_adaptive_ensemble.csv
# - horizon_adaptive_weights.csv
# ============================================================

!pip install lightgbm xgboost catboost -q

import os, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

SEED = 42
np.random.seed(SEED)
start_time = time.time()

PROJECT_DIR = "/content/drive/MyDrive/DENGUE_BRASIL"
BENCH_FILE = os.path.join(PROJECT_DIR, "BENCHMARK", "dataset_benchmark_v1.parquet")
MASTER_FILE = os.path.join(PROJECT_DIR, "DATASETS_MASTER_MULTIMODAL", "BraDengue-MM.parquet")
OUT_DIR = os.path.join(PROJECT_DIR, "BENCHMARK", "EXTENDED_EXPERIMENTS")
os.makedirs(OUT_DIR, exist_ok=True)

bench = pd.read_parquet(BENCH_FILE)
targets_full = pd.read_parquet(MASTER_FILE, columns=["codigo_ibge", "anio", "semana", "p_inc100k"])
targets_full = targets_full.sort_values(["codigo_ibge", "anio", "semana"]).reset_index(drop=True)

HORIZONS = [1, 2, 4, 8]
for h in HORIZONS:
    targets_full[f"target_h{h}"] = targets_full.groupby("codigo_ibge")["p_inc100k"].shift(-(h - 1))

bench = bench.merge(targets_full.drop(columns=["p_inc100k"]), on=["codigo_ibge", "anio", "semana"], how="left")
del targets_full

bench["persist_base"] = (bench["casos_lag_1"] / (bench["pop"] + 1e-6)) * 100000.0
bench = bench.dropna(subset=[f"target_h{h}" for h in HORIZONS] + ["persist_base"]).sort_values(["anio", "semana", "codigo_ibge"]).reset_index(drop=True)

exclude_cols = {
    "codigo_ibge", "municipio", "estado", "data_iniSE", "SE", "anio", "semana",
    "p_inc100k", "casos_est", "casos_est_min", "casos_est_max", "casprov",
    "nivel_inc", "nivel", "casos", "Rt", "p_rt1", "transmissao", "receptivo",
    "notif_accum_year", "persist_base", "dominant_landcover"
} | {f"target_h{h}" for h in HORIZONS}

features = [c for c in bench.select_dtypes(include=np.number).columns if c not in exclude_cols]
folds = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]

def fit_simplex_weights(P_val, y_val):
    k = P_val.shape[1]
    res = minimize(
        lambda w: np.mean((y_val - P_val @ w) ** 2),
        x0=np.ones(k) / k,
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    )
    return res.x

import shutil
has_gpu = shutil.which("nvidia-smi") is not None
print(f"Hardware detected: {'GPU' if has_gpu else 'CPU'}")

fold_res = {(h, m): [] for h in HORIZONS for m in ["HorizonAdaptiveEnsemble", "EqualGBDT_Ensemble", "LightGBM", "CatBoost", "XGBoost", "NaivePersistence"]}
w_folds = {h: [] for h in HORIZONS}

for fold_id, (tr_end, te_yr) in enumerate(folds, 1):
    t0 = time.time()
    sub_tr_mask = bench["anio"] < tr_end
    val_mask = bench["anio"] == tr_end
    tr_full_mask = bench["anio"] <= tr_end
    te_mask = bench["anio"] == te_yr

    imp_v = SimpleImputer(strategy="median")
    X_sub = imp_v.fit_transform(bench.loc[sub_tr_mask, features].astype(np.float32))
    X_val = imp_v.transform(bench.loc[val_mask, features].astype(np.float32))

    imp_f = SimpleImputer(strategy="median")
    X_tr = imp_f.fit_transform(bench.loc[tr_full_mask, features].astype(np.float32))
    X_te = imp_f.transform(bench.loc[te_mask, features].astype(np.float32))

    p_nai_val = bench.loc[val_mask, "persist_base"].values.astype(np.float32)
    p_nai_te = bench.loc[te_mask, "persist_base"].values.astype(np.float32)

    xgb_kw = dict(tree_method="hist", device="cuda") if has_gpu else dict(tree_method="hist", n_jobs=-1)
    cat_kw = dict(task_type="GPU", iterations=200, depth=6) if has_gpu else dict(thread_count=-1, iterations=100, depth=5, border_count=64)

    for h in HORIZONS:
        target_col = f"target_h{h}"
        y_sub = bench.loc[sub_tr_mask, target_col].values.astype(np.float32)
        y_val = bench.loc[val_mask, target_col].values.astype(np.float32)
        y_tr = bench.loc[tr_full_mask, target_col].values.astype(np.float32)
        y_te = bench.loc[te_mask, target_col].values.astype(np.float32)

        m_lgb_v = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, max_bin=127, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1).fit(X_sub, y_sub)
        m_xgb_v = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=SEED, **xgb_kw).fit(X_sub, y_sub)
        m_cat_v = CatBoostRegressor(learning_rate=0.05, random_seed=SEED, verbose=0, **cat_kw).fit(X_sub, y_sub)


        P_val = np.column_stack([m_lgb_v.predict(X_val), m_cat_v.predict(X_val), m_xgb_v.predict(X_val), p_nai_val])
        w_opt = fit_simplex_weights(P_val, y_val)
        w_folds[h].append(w_opt)

        p_lgb = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, max_bin=127, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1).fit(X_tr, y_tr).predict(X_te)
        p_xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=SEED, **xgb_kw).fit(X_tr, y_tr).predict(X_te)
        p_cat = CatBoostRegressor(learning_rate=0.05, random_seed=SEED, verbose=0, **cat_kw).fit(X_tr, y_tr).predict(X_te)


        P_te = np.column_stack([p_lgb, p_cat, p_xgb, p_nai_te])
        p_hae = P_te @ w_opt
        p_eq = (p_lgb + p_cat + p_xgb) / 3.0

        for name, pred in [("HorizonAdaptiveEnsemble", p_hae), ("EqualGBDT_Ensemble", p_eq), ("LightGBM", p_lgb), ("CatBoost", p_cat), ("XGBoost", p_xgb), ("NaivePersistence", p_nai_te)]:
            fold_res[(h, name)].append((mean_absolute_error(y_te, pred), np.sqrt(mean_squared_error(y_te, pred)), r2_score(y_te, pred)))

    print(f"Fold {fold_id}/5 done in {(time.time()-t0):.1f}s")

ens_rows, weight_rows = [], []
for h in HORIZONS:
    for name in ["HorizonAdaptiveEnsemble", "EqualGBDT_Ensemble", "LightGBM", "CatBoost", "XGBoost", "NaivePersistence"]:
        arr = np.array(fold_res[(h, name)])
        ens_rows.append({
            "Horizon_Weeks": h, "Model": name,
            "MAE_Mean": arr[:, 0].mean(), "MAE_SD": arr[:, 0].std(),
            "RMSE_Mean": arr[:, 1].mean(), "RMSE_SD": arr[:, 1].std(),
            "R2_Mean": arr[:, 2].mean(), "R2_SD": arr[:, 2].std()
        })
    w_mean = np.mean(w_folds[h], axis=0)
    weight_rows.append({
        "Horizon_Weeks": h,
        "w_LightGBM": w_mean, "w_CatBoost": w_mean,
        "w_XGBoost": w_mean, "w_NaivePersistence": w_mean
    })

ens_df = pd.DataFrame(ens_rows)
w_df = pd.DataFrame(weight_rows)

ens_df.to_csv(os.path.join(OUT_DIR, "horizon_adaptive_ensemble.csv"), index=False)
w_df.to_csv(os.path.join(OUT_DIR, "horizon_adaptive_weights.csv"), index=False)

print("\n=== TABLE: HORIZON-ADAPTIVE ENSEMBLE PERFORMANCE ===")
print(ens_df.to_string(index=False))
print("\n=== TABLE: LEARNED HORIZON-ADAPTIVE WEIGHTS ===")
print(w_df.to_string(index=False))
print(f"\n[✓] Completed in {(time.time() - start_time)/60:.2f} min.")
