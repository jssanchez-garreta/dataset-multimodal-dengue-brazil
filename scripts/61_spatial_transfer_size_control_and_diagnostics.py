"""
================================================================================
Script: 61_spatial_transfer_size_control_and_diagnostics.py
Project: BraDengue-MM — Multimodal Dengue Surveillance & Forecasting in Brazil
================================================================================
Purpose:
    Strengthens the Leave-Macro-Region-Out spatial transferability audit
    (2010–2023 training -> 2024–2025 testing) at h = 1 and h = 4 weeks by:
      1. Adding a training-size-matched out-of-region control (`SizeMatched_ZS`)
         subsampled to the exact sample count of each local `Within` model to
         disentangle sample-size effects from cross-biome generalization.
      2. Reporting MAE, RMSE, R², and municipality-level block-bootstrap 95%
         confidence intervals (B = 500 municipal resamples).
      3. Evaluating prediction error across thermal terciles (`tempmed`) in the
         subtropical South at h = 4 weeks to verify thermal-threshold divergence.

Inputs:
    - FULL_FILE: Complete nationwide multimodal panel (4,634,540 records × 87 cols):
      `BraDengue-MM.parquet`
    - BENCH_FILE: Year-stratified 800,000-record benchmark subset:
      `dataset_benchmark_v1.parquet`

Outputs:
    - Console tables of Within-Region, Full Zero-Shot, and Size-Matched Zero-Shot
      metrics (MAE, RMSE, R² with 95% municipal bootstrap CI) at h = 1 and h = 4,
      plus South h = 4 error across temperature terciles.
    - CSV files saved to `OUT_DIR/spatial_transfer_size_controlled.csv` and
      `OUT_DIR/south_h4_thermal_diagnostic.csv`.
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

reg_map = {"1": "North", "2": "Northeast", "3": "Southeast", "4": "South", "5": "Central-West"}
bench["Region"] = bench["codigo_ibge"].astype(str).str.zfill(7).str.slice(0, 1).map(reg_map)

se_sorted = np.sort(bench["SE"].unique())
se_to_idx = {se: i for i, se in enumerate(se_sorted)}
bench["t_idx"] = bench["SE"].map(se_to_idx).astype(np.int32)
T_2023_end = bench.loc[bench["anio"] == 2023, "t_idx"].max()

LEAKAGE = {"Unnamed: 0", "codigo_ibge", "municipio", "estado", "SE", "anio", "semana", "data_iniSE", "casos_est", "casos_est_min", "casos_est_max", "nivel", "nivel_inc", "casprov", "casos", "Rt", "p_rt1", "transmissao", "receptivo", "notif_accum_year", "dominant_land_cover", "Region", "t_idx", "target_h1", "target_h4"}
features = [c for c in bench.columns if c not in LEAKAGE and pd.api.types.is_numeric_dtype(bench[c])]

def muni_bootstrap_r2(munis, y_true, y_pred, B=500, seed=42):
    rng = np.random.default_rng(seed)
    df_tmp = pd.DataFrame({"m": munis, "y": y_true, "p": y_pred})
    uniq_m = df_tmp["m"].unique()
    groups = {m: g[["y", "p"]].values for m, g in df_tmp.groupby("m")}
    vals = []
    for _ in range(B):
        samp = rng.choice(uniq_m, size=len(uniq_m), replace=True)
        arr = np.vstack([groups[m] for m in samp])
        vals.append(r2_score(arr[:, 0], arr[:, 1]))
    return np.percentile(vals, 2.5), np.percentile(vals, 97.5)

regions = ["North", "Northeast", "Southeast", "South", "Central-West"]
rng = np.random.default_rng(42)
rows = []
t0 = time.time()

for h in (1, 4):
    sub = bench.dropna(subset=[f"target_h{h}"]).reset_index(drop=True)
    tr_base = sub["t_idx"] <= (T_2023_end - h)
    te_base = sub["anio"] >= 2024
    print(f"\n=== SPATIAL TRANSFERABILITY WITH SIZE CONTROL & 95% CI (h = {h}) ===")

    for reg in regions:
        te_mask = te_base & (sub["Region"] == reg)
        tr_within = tr_base & (sub["Region"] == reg)
        tr_out = tr_base & (sub["Region"] != reg)

        n_within = tr_within.sum()
        out_indices = np.where(tr_out)[0]
        matched_idx = rng.choice(out_indices, size=min(n_within, len(out_indices)), replace=False)
        tr_matched = np.zeros(len(sub), dtype=bool)
        tr_matched[matched_idx] = True

        X_te_raw = sub.loc[te_mask, features].astype(np.float32)
        y_te = sub.loc[te_mask, f"target_h{h}"].values
        munis_te = sub.loc[te_mask, "codigo_ibge"].values

        res_reg = {}
        for mode, mask in [("Within", tr_within), ("Full_ZeroShot", tr_out), ("SizeMatched_ZS", tr_matched)]:
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(sub.loc[mask, features].astype(np.float32))
            X_te = imp.transform(X_te_raw)
            y_tr = sub.loc[mask, f"target_h{h}"].values
            pred = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, max_bin=127, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1).fit(X_tr, y_tr).predict(X_te)
            r2 = r2_score(y_te, pred)
            mae = mean_absolute_error(y_te, pred)
            rmse = np.sqrt(mean_squared_error(y_te, pred))
            lo, hi = muni_bootstrap_r2(munis_te, y_te, pred, B=300)
            res_reg[mode] = (r2, lo, hi, mae, rmse, pred)

        w_r2, w_lo, w_hi, w_mae, w_rmse, _ = res_reg["Within"]
        f_r2, f_lo, f_hi, f_mae, f_rmse, f_pred = res_reg["Full_ZeroShot"]
        m_r2, m_lo, m_hi, m_mae, m_rmse, _ = res_reg["SizeMatched_ZS"]
        print(f"{reg:13s} (N_tr={n_within:6d}) | Within R2={w_r2:.4f} [{w_lo:.3f},{w_hi:.3f}] MAE={w_mae:.2f} | FullZS R2={f_r2:.4f} [{f_lo:.3f},{f_hi:.3f}] MAE={f_mae:.2f} | MatchedZS R2={m_r2:.4f} [{m_lo:.3f},{m_hi:.3f}] MAE={m_mae:.2f}")
        rows.append({
            "Horizon": h, "Region": reg, "N_Train_Within": n_within, "N_Test": len(y_te),
            "Within_R2": w_r2, "Within_CI_Lo": w_lo, "Within_CI_Hi": w_hi, "Within_MAE": w_mae, "Within_RMSE": w_rmse,
            "FullZS_R2": f_r2, "FullZS_CI_Lo": f_lo, "FullZS_CI_Hi": f_hi, "FullZS_MAE": f_mae, "FullZS_RMSE": f_rmse,
            "MatchedZS_R2": m_r2, "MatchedZS_CI_Lo": m_lo, "MatchedZS_CI_Hi": m_hi, "MatchedZS_MAE": m_mae, "MatchedZS_RMSE": m_rmse,
        })

        # Thermal diagnostic in South at h = 4
        if h == 4 and reg == "South":
            temp_te = sub.loc[te_mask, "tempmed"].values
            valid_t = ~np.isnan(temp_te)
            q33, q66 = np.nanpercentile(temp_te, [33.3, 66.7])
            bins = np.digitize(temp_te[valid_t], [q33, q66])
            diag_rows = []
            print(f"\n=== SOUTH h=4 THERMAL REGIME DIAGNOSTIC (tempmed terciles: <{q33:.1f}C, {q33:.1f}-{q66:.1f}C, >{q66:.1f}C) ===")
            for b_id, b_lbl in enumerate(["Cool Tercile (<33%)", "Mild Tercile (33-67%)", "Warm Tercile (>67%)"]):
                idx_b = bins == b_id
                yb = y_te[valid_t][idx_b]
                pb_zs = f_pred[valid_t][idx_b]
                pb_in = res_reg["Within"][5][valid_t][idx_b]
                print(f"  {b_lbl:22s} | MeanInc={yb.mean():6.2f} | Within MAE={mean_absolute_error(yb, pb_in):5.2f} | ZeroShot MAE={mean_absolute_error(yb, pb_zs):5.2f} | ZS Bias={(pb_zs - yb).mean():+6.2f}")
                diag_rows.append({"Tercile": b_lbl, "Mean_Incidence": yb.mean(), "Within_MAE": mean_absolute_error(yb, pb_in), "ZeroShot_MAE": mean_absolute_error(yb, pb_zs), "ZeroShot_Bias": (pb_zs - yb).mean()})
            pd.DataFrame(diag_rows).to_csv(os.path.join(OUT_DIR, "south_h4_thermal_diagnostic.csv"), index=False)

pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "spatial_transfer_size_controlled.csv"), index=False)
print(f"\nCompleted in {(time.time()-t0):.1f}s")