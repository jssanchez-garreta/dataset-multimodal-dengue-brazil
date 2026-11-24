"""
================================================================================
Script: 56_grouped_modality_attribution_robustness.py
Project: BraDengue-MM — Multimodal Dengue Surveillance & Forecasting in Brazil
================================================================================
Purpose:
    Assesses the robustness of horizon-dependent modality importance shifts
    across expanding-window temporal folds (2021–2025) at short (h = 1 week)
    and medium (h = 8 weeks) horizons. Complements split-count profiles with
    two multicollinearity-aware diagnostics grouped across five functional
    data modalities:
      1. Fold-averaged LightGBM total-gain shares (%, mean ± SD across folds).
      2. Out-of-sample grouped permutation importance (ΔR², mean ± SD across
         folds, averaged over 3 random group permutations per fold).

Inputs:
    - FULL_FILE: Complete nationwide multimodal panel (4,634,540 records × 87 cols)
      used to compute causal future targets (target_h1, target_h8) prior to
      benchmark alignment:
      `BraDengue-MM.parquet`
    - BENCH_FILE: Year-stratified 800,000-record benchmark subset:
      `dataset_benchmark_v1.parquet`

Outputs:
    - Console summary of fold-averaged (mean ± SD) grouped Gain Share (%) and
      Grouped Permutation ΔR² at h = 1 and h = 8 weeks (reported in
      Supplementary Table S7).
    - CSV file saved to `OUT_DIR/grouped_modality_attribution_robustness.csv`.
================================================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor

FULL_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL/BraDengue-MM.parquet"
BENCH_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/dataset_benchmark_v1.parquet"
OUT_DIR = "/content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/EXTENDED_EXPERIMENTS"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Compute causal future targets on the continuous 4.63M panel first
full_t = (
    pd.read_parquet(FULL_FILE, columns=["codigo_ibge", "SE", "p_inc100k"])
    .sort_values(["codigo_ibge", "SE"])
    .reset_index(drop=True)
)
for h in (1, 8):
    full_t[f"target_h{h}"] = full_t.groupby("codigo_ibge")["p_inc100k"].shift(-h).astype(np.float32)

bench = pd.read_parquet(BENCH_FILE).merge(
    full_t[["codigo_ibge", "SE", "target_h1", "target_h8"]],
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
    "dominant_land_cover", "t_idx", "target_h1", "target_h8",
}
features = [c for c in bench.columns if c not in LEAKAGE and pd.api.types.is_numeric_dtype(bench[c])]


def get_group(col: str) -> str:
    if col == "p_inc100k" or col.startswith("casos_") or col.startswith("Rt_") or col == "mes":
        return "Epidemiological Memory & Seasonality"
    elif col in ("nino34", "soi"):
        return "ENSO Teleconnections"
    elif col.startswith(("NDVI", "EVI", "LST")):
        return "MODIS Remote Sensing"
    elif col.startswith(("temp", "umid", "precip", "dias_lluvia")):
        return "Surface Meteo & CHIRPS"
    else:
        return "Static Context (Demo/Topo/LandCover/Sanit)"


groups = {}
for f in features:
    groups.setdefault(get_group(f), []).append(f)

folds = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
rng = np.random.default_rng(42)
rows = []

for h in (1, 8):
    sub = bench.dropna(subset=[f"target_h{h}"]).reset_index(drop=True)
    perm_drops = {g: [] for g in groups}
    gain_shares = {g: [] for g in groups}

    for tr_end, te_yr in folds:
        T_tr_end = year_max_t[tr_end]
        tr = sub["t_idx"] <= (T_tr_end - h)
        te = sub["anio"] == te_yr
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(sub.loc[tr, features].astype(np.float32))
        X_te = imp.transform(sub.loc[te, features].astype(np.float32))
        y_tr = sub.loc[tr, f"target_h{h}"].values
        y_te = sub.loc[te, f"target_h{h}"].values

        model = LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=15, max_bin=127,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
        ).fit(X_tr, y_tr)
        base_r2 = r2_score(y_te, model.predict(X_te))

        gains = model.booster_.feature_importance(importance_type="gain")
        tot_gain = gains.sum()
        for g, cols in groups.items():
            idxs = [features.index(c) for c in cols]
            gain_shares[g].append(100.0 * gains[idxs].sum() / tot_gain)
            drops = []
            for _ in range(3):
                X_perm = X_te.copy()
                perm_idx = rng.permutation(len(X_te))
                X_perm[:, idxs] = X_perm[perm_idx][:, idxs]
                drops.append(base_r2 - r2_score(y_te, model.predict(X_perm)))
            perm_drops[g].append(np.mean(drops))

    print(f"\n=== GROUPED ATTRIBUTION AT h = {h} WEEKS (Mean ± SD across 5 folds) ===")
    for g in groups:
        gd = np.array(perm_drops[g])
        gs = np.array(gain_shares[g])
        print(f"{g:42s} | Perm ΔR2 = {gd.mean():.4f} ± {gd.std():.4f} | Gain Share = {gs.mean():.2f}% ± {gs.std():.2f}%")
        rows.append({
            "Horizon_Weeks": h,
            "Modality_Group": g,
            "Gain_Share_Mean_Pct": gs.mean(),
            "Gain_Share_SD_Pct": gs.std(),
            "Perm_Delta_R2_Mean": gd.mean(),
            "Perm_Delta_R2_SD": gd.std(),
        })

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "grouped_modality_attribution_robustness.csv"), index=False)