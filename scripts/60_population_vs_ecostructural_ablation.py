"""
================================================================================
Script: 60_population_vs_ecostructural_ablation.py
Project: BraDengue-MM — Multimodal Dengue Surveillance & Forecasting in Brazil
================================================================================
Purpose:
    Disentangles population-denominator scaling effects from physical
    eco-structural covariates (topography, land cover, sanitation) in the
    progressive modality ablation hierarchy across 5 expanding-window folds
    (2021–2025) at short (h = 1 week) and medium (h = 4 weeks) horizons under
    strict boundary trimming (t <= T_train,end - h). Compares:
      1. M4: Dynamic epidemiological, meteorological, CHIRPS, MODIS, and ENSO streams
      2. M4b: M4 + Population Denominators & Demographic structure
      3. M5: Full Multimodal (+ Topography, ESA WorldCover, and Sanitation)

Inputs:
    - FULL_FILE: Complete nationwide multimodal panel (4,634,540 records × 87 cols)
      used to compute causal future targets (`target_h1`, `target_h4`):
      `BraDengue-MM.parquet`
    - BENCH_FILE: Year-stratified 800,000-record benchmark subset:
      `dataset_benchmark_v1.parquet`

Outputs:
    - Console summary of out-of-sample MAE, RMSE, and R² (mean ± SD across folds)
      for M4, M4b, and M5 at h = 1 and h = 4 weeks.
    - CSV file saved to `OUT_DIR/population_vs_ecostructural_ablation.csv`.
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

FULL_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL/BraDengue-MM.parquet"
BENCH_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/dataset_benchmark_v1.parquet"
OUT_DIR = "/content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/EXTENDED_EXPERIMENTS"
os.makedirs(OUT_DIR, exist_ok=True)

full_t = pd.read_parquet(FULL_FILE, columns=["codigo_ibge", "SE", "p_inc100k"]).sort_values(["codigo_ibge", "SE"]).reset_index(drop=True)
for h in (1, 4):
    full_t[f"target_h{h}"] = full_t.groupby("codigo_ibge")["p_inc100k"].shift(-h).astype(np.float32)

bench = pd.read_parquet(BENCH_FILE).merge(full_t[["codigo_ibge", "SE", "target_h1", "target_h4"]], on=["codigo_ibge", "SE"], how="left")
del full_t

se_sorted = np.sort(bench["SE"].unique())
se_to_idx = {se: i for i, se in enumerate(se_sorted)}
bench["t_idx"] = bench["SE"].map(se_to_idx).astype(np.int32)
year_max_t = bench.groupby("anio")["t_idx"].max().to_dict()

LEAKAGE = {"Unnamed: 0", "codigo_ibge", "municipio", "estado", "SE", "anio", "semana", "data_iniSE", "casos_est", "casos_est_min", "casos_est_max", "nivel", "nivel_inc", "casprov", "casos", "Rt", "p_rt1", "transmissao", "receptivo", "notif_accum_year", "dominant_land_cover", "t_idx", "target_h1", "target_h4"}
all_feats = [c for c in bench.columns if c not in LEAKAGE and pd.api.types.is_numeric_dtype(bench[c])]

# Define M4 (Epidemiology + Meteo/CHIRPS + MODIS + ENSO)
m_epi = [c for c in all_feats if c == "p_inc100k" or c.startswith(("casos_", "Rt_")) or c == "mes"]
m_met = [c for c in all_feats if c.startswith(("temp", "umid", "precip", "dias_lluvia"))]
m_mod = [c for c in all_feats if c.startswith(("NDVI", "EVI", "LST"))]
m_ens = [c for c in all_feats if c in ("nino34", "soi")]
m4_feats = m_epi + m_met + m_mod + m_ens

# M4b: M4 + Population Denominators & Demography (isolating denominator scaling)
m_pop = [c for c in all_feats if c in ("pop", "population_total", "population_density", "male_population", "female_population", "sex_ratio", "median_age", "aging_index", "elderly_pct", "youth_pct")]
m4b_feats = m4_feats + [c for c in m_pop if c not in m4_feats]

folds = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
rows = []
t0 = time.time()
for h in (1, 4):
    sub = bench.dropna(subset=[f"target_h{h}"]).reset_index(drop=True)
    for name, fset in [("M4 (Dynamic Clim/EO)", m4_feats), ("M4b (+ Population/Demo)", m4b_feats), ("M5 (+ Topo/Land/Sanit)", all_feats)]:
        scores = []
        for tr_end, te_yr in folds:
            T_tr_end = year_max_t[tr_end]
            tr = sub["t_idx"] <= (T_tr_end - h)
            te = sub["anio"] == te_yr
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(sub.loc[tr, fset].astype(np.float32))
            X_te = imp.transform(sub.loc[te, fset].astype(np.float32))
            y_tr = sub.loc[tr, f"target_h{h}"].values
            y_te = sub.loc[te, f"target_h{h}"].values
            pred = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, max_bin=127, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1).fit(X_tr, y_tr).predict(X_te)
            scores.append((mean_absolute_error(y_te, pred), np.sqrt(mean_squared_error(y_te, pred)), r2_score(y_te, pred)))
        arr = np.array(scores)
        print(f"h={h} | {name:25s} ({len(fset):2d} vars) | MAE={arr[:,0].mean():.2f} | RMSE={arr[:,1].mean():.2f} | R2={arr[:,2].mean():.4f}±{arr[:,2].std():.4f}")
        rows.append({
            "Horizon_Weeks": h,
            "Configuration": name,
            "Num_Features": len(fset),
            "MAE_Mean": arr[:,0].mean(),
            "RMSE_Mean": arr[:,1].mean(),
            "R2_Mean": arr[:,2].mean(),
            "R2_SD": arr[:,2].std(),
        })
print(f"Done in {(time.time()-t0):.1f}s")
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "population_vs_ecostructural_ablation.csv"), index=False)