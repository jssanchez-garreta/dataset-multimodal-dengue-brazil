"""
================================================================================
Script: 57_sanitation_mnar_and_regional_completeness.py
Project: BraDengue-MM — Multimodal Dengue Surveillance & Forecasting in Brazil
================================================================================
Purpose:
    Audits macro-regional data completeness and tests whether spatially
    structured (MNAR) missingness in municipal sanitation registries drives
    predictive gains in the full multimodal model (M5). Specifically:
      1. Computes macro-regional completeness (%) across all 4,634,540 records
         for municipal sanitation indicators (`water_supply_pct`, `sewage_pct`,
         `garbage_collection_pct`) and key MODIS/CHIRPS environmental streams
         (reported in Supplementary Table S6).
      2. Performs a 5-fold expanding-window sensitivity analysis (2021–2025)
         at h = 1 and h = 4 weeks comparing Full M5 (67 features), M5 without
         sanitation variables (64 features), and M5 augmented with explicit
         binary missingness indicators (70 features).

Inputs:
    - FULL_FILE: Complete nationwide multimodal panel (4,634,540 records × 87 cols):
      `BraDengue-MM.parquet`
    - BENCH_FILE: Year-stratified 800,000-record benchmark subset:
      `dataset_benchmark_v1.parquet`

Outputs:
    - Console tables of macro-regional completeness (%) and M5 sanitation
      sensitivity metrics (MAE, RMSE, R² mean ± SD across folds).
    - CSV files saved to `OUT_DIR/macro_regional_completeness.csv` and
      `OUT_DIR/sanitation_mnar_sensitivity.csv`.
================================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

FULL_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL/BraDengue-MM.parquet"
BENCH_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/dataset_benchmark_v1.parquet"
OUT_DIR = "/content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/EXTENDED_EXPERIMENTS"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Macro-regional completeness audit on full panel (4.63M)
audit_cols = [
    "codigo_ibge", "water_supply_pct", "sewage_pct", "garbage_collection_pct",
    "NDVI_mean", "LST_Day_mean", "LST_Night_mean", "precip_total_semana",
]
full = pd.read_parquet(FULL_FILE, columns=audit_cols)
reg_map = {"1": "North", "2": "Northeast", "3": "Southeast", "4": "South", "5": "Central-West"}
full["Region"] = full["codigo_ibge"].astype(str).str.zfill(7).str.slice(0, 1).map(reg_map)

print("=== MACRO-REGIONAL COMPLETENESS (%) ===")
comp_reg = (
    full.groupby("Region")[audit_cols[1:]]
    .apply(lambda g: g.notna().mean() * 100)
    .loc[["North", "Northeast", "Southeast", "South", "Central-West"]]
)
print(comp_reg.round(2).to_string())
comp_reg.round(2).to_csv(os.path.join(OUT_DIR, "macro_regional_completeness.csv"))
del full

# 2. Sensitivity analysis on M5 (h=1 and h=4) under strict boundary trimming
full_t = (
    pd.read_parquet(FULL_FILE, columns=["codigo_ibge", "SE", "p_inc100k"])
    .sort_values(["codigo_ibge", "SE"])
    .reset_index(drop=True)
)
for h in (1, 4):
    full_t[f"target_h{h}"] = full_t.groupby("codigo_ibge")["p_inc100k"].shift(-h).astype(np.float32)

bench = pd.read_parquet(BENCH_FILE).merge(
    full_t[["codigo_ibge", "SE", "target_h1", "target_h4"]],
    on=["codigo_ibge", "SE"],
    how="left",
)
del full_t

se_sorted = np.sort(bench["SE"].unique())
se_to_idx = {se: i for i, se in enumerate(se_sorted)}
bench["t_idx"] = bench["SE"].map(se_to_idx).astype(np.int32)
year_max_t = bench.groupby("anio")["t_idx"].max().to_dict()

LEAKAGE = {
    "Unnamed: 0", "codigo_ibge", "municipio", "estado", "SE", "anio", "semana", "data_iniSE",
    "casos_est", "casos_est_min", "casos_est_max", "nivel", "nivel_inc",
    "casprov", "casos", "Rt", "p_rt1", "transmissao", "receptivo", "notif_accum_year",
    "dominant_land_cover", "t_idx", "target_h1", "target_h4",
}
feats_full = [c for c in bench.columns if c not in LEAKAGE and pd.api.types.is_numeric_dtype(bench[c])]
san_cols = ["water_supply_pct", "sewage_pct", "garbage_collection_pct"]
feats_nosan = [c for c in feats_full if c not in san_cols]

for c in san_cols:
    bench[f"{c}_isna"] = bench[c].isna().astype(np.float32)
feats_ind = feats_full + [f"{c}_isna" for c in san_cols]

folds = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
sens_rows = []
print("\n=== SANITATION SENSITIVITY ANALYSIS (M5, h=1 & h=4) ===")
for h in (1, 4):
    sub = bench.dropna(subset=[f"target_h{h}"]).reset_index(drop=True)
    for cfg_name, fset in [
        ("M5 Full (67 vars)", feats_full),
        ("M5 No-Sanitation (64 vars)", feats_nosan),
        ("M5 + Missing Flags (70 vars)", feats_ind),
    ]:
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
            pred = LGBMRegressor(
                n_estimators=200, learning_rate=0.05, num_leaves=15, max_bin=127,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
            ).fit(X_tr, y_tr).predict(X_te)
            scores.append((
                mean_absolute_error(y_te, pred),
                np.sqrt(mean_squared_error(y_te, pred)),
                r2_score(y_te, pred),
            ))
        arr = np.array(scores)
        print(f"h={h} | {cfg_name:28s} | MAE={arr[:,0].mean():.2f} | RMSE={arr[:,1].mean():.2f} | R2={arr[:,2].mean():.4f}±{arr[:,2].std():.4f}")
        sens_rows.append({
            "Horizon_Weeks": h,
            "Configuration": cfg_name,
            "Num_Features": len(fset),
            "MAE_Mean": arr[:,0].mean(),
            "RMSE_Mean": arr[:,1].mean(),
            "R2_Mean": arr[:,2].mean(),
            "R2_SD": arr[:,2].std(),
        })

pd.DataFrame(sens_rows).to_csv(os.path.join(OUT_DIR, "sanitation_mnar_sensitivity.csv"), index=False)
