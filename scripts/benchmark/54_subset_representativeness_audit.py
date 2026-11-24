"""
================================================================================
Script: 54_subset_representativeness_audit.py
Project: BraDengue-MM — Multimodal Dengue Surveillance & Forecasting in Brazil
================================================================================
Purpose:
    Audits spatial and epidemiological representativeness of the year-stratified
    800,000-record benchmark subset (`dataset_benchmark_v1.parquet`) against the
    complete 4,634,540-record nationwide panel (`BraDengue-MM.parquet`).
    Computes macro-regional observation shares (%) and weekly incidence
    (`p_inc100k`) distribution moments and quantiles (reported in Table 3).

Inputs:
    - FULL_FILE: Complete nationwide multimodal panel (4,634,540 records × 87 cols):
      `BraDengue-MM.parquet`
    - BENCH_FILE: Year-stratified 800,000-record benchmark subset:
      `dataset_benchmark_v1.parquet`

Outputs:
    - Console comparison of macro-regional shares (%) and incidence distribution
      statistics (Mean, SD, Zero-incidence %, P50, P75, P90, P99).
    - CSV summaries saved to `OUT_DIR/subset_macro_region_shares.csv` and
      `OUT_DIR/subset_incidence_distribution.csv`.
================================================================================
"""

import os
import numpy as np
import pandas as pd

FULL_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL/BraDengue-MM.parquet"
BENCH_FILE = "/content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/dataset_benchmark_v1.parquet"
OUT_DIR = "/content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/EXTENDED_EXPERIMENTS"
os.makedirs(OUT_DIR, exist_ok=True)

full = pd.read_parquet(FULL_FILE, columns=["codigo_ibge", "p_inc100k"])
bench = pd.read_parquet(BENCH_FILE, columns=["codigo_ibge", "p_inc100k"])

reg_map = {"1": "North", "2": "Northeast", "3": "Southeast", "4": "South", "5": "Central-West"}
full["region"] = full["codigo_ibge"].astype(str).str.zfill(7).str.slice(0, 1).map(reg_map)
bench["region"] = bench["codigo_ibge"].astype(str).str.zfill(7).str.slice(0, 1).map(reg_map)

print("=== MACRO-REGION SHARES (%) ===")
reg_comp = pd.DataFrame({
    "Full_Pct": full["region"].value_counts(normalize=True) * 100,
    "Bench_Pct": bench["region"].value_counts(normalize=True) * 100,
}).loc[["North", "Northeast", "Southeast", "South", "Central-West"]]
print(reg_comp.round(2))
reg_comp.round(2).to_csv(os.path.join(OUT_DIR, "subset_macro_region_shares.csv"))

print("\n=== INCIDENCE DISTRIBUTION (p_inc100k) ===")
dist_rows = []
for name, df in [("Full (4.63M)", full), ("Benchmark (800k)", bench)]:
    s = df["p_inc100k"].dropna()
    zero_pct = (s == 0).mean() * 100
    q50, q75, q90, q99 = np.percentile(s, [50, 75, 90, 99])
    print(f"{name}: Mean={s.mean():.2f}, SD={s.std():.2f}, Zero%={zero_pct:.2f}%, P50={q50:.2f}, P75={q75:.2f}, P90={q90:.2f}, P99={q99:.2f}")
    dist_rows.append({
        "Dataset": name,
        "Mean": round(s.mean(), 2),
        "SD": round(s.std(), 2),
        "Zero_Pct": round(zero_pct, 2),
        "P50": round(q50, 2),
        "P75": round(q75, 2),
        "P90": round(q90, 2),
        "P99": round(q99, 2),
    })

pd.DataFrame(dist_rows).to_csv(os.path.join(OUT_DIR, "subset_incidence_distribution.csv"), index=False)