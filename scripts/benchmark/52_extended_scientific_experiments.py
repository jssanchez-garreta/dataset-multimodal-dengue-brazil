# ============================================================
# 52_extended_scientific_experiments.py
#
# PROJECT:
# Early Detection of Dengue Outbreaks in Brazil
#
# OBJECTIVE:
# Execute three complementary scientific experiments on BraDengue-MM
# to evaluate prospective forecasting decay, multimodal data fusion
# gains, and cross-biome spatial generalizability:
#   1) Multi-Horizon Prospective Forecasting (h in {1, 2, 4, 8} weeks)
#      across all 9 benchmark models (LightGBM, CatBoost, Naive
#      Persistence, XGBoost, Cascade Forest, 5-Layer MLP, Random
#      Forest, Ridge Regression, and LinearSVR) under 5-fold expanding-
#      window temporal validation (2021-2025), tracking modality split-
#      importance shifts across horizons.
#   2) Systematic Modality Ablation Study (M1 to M5) at short (h = 1)
#      and medium-term (h = 4) horizons to quantify incremental gains
#      from meteorology/CHIRPS, MODIS Earth observation, ENSO indices,
#      and static socio-environmental context over epidemiology alone.
#   3) Zero-Shot Spatial Generalization (Leave-Macro-Region-Out)
#      across Brazil's 5 official macro-regions (North, Northeast,
#      Southeast, South, and Central-West) at h = 1 and h = 4 weeks.
#
# INPUTS:
# - /content/drive/MyDrive/DENGUE_BRASIL/BENCHMARK/dataset_benchmark_v1.parquet
# - /content/drive/MyDrive/DENGUE_BRASIL/DATASETS_MASTER_MULTIMODAL/BraDengue-MM.parquet
#   (lightweight 4-column index lookup for exact future targets t+h)
#
# OUTPUTS (saved in DENGUE_BRASIL/BENCHMARK/EXTENDED_EXPERIMENTS/):
# - multi_horizon_9models.csv
# - modality_share_horizon.csv
# - modality_ablation.csv
# - spatial_transfer.csv
# - extended_scientific_results.xlsx
# - Figure_7_MultiHorizon_ModalityShift.png
# ============================================================

!pip install lightgbm xgboost catboost -q

import os, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
start_time = time.time()

PROJECT_DIR = "/content/drive/MyDrive/DENGUE_BRASIL"
BENCH_FILE = os.path.join(PROJECT_DIR, "BENCHMARK", "dataset_benchmark_v1.parquet")
MASTER_FILE = os.path.join(PROJECT_DIR, "DATASETS_MASTER_MULTIMODAL", "BraDengue-MM.parquet")
OUT_DIR = os.path.join(PROJECT_DIR, "BENCHMARK", "EXTENDED_EXPERIMENTS")
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading dataset_benchmark_v1.parquet (800,000 rows)...")
bench = pd.read_parquet(BENCH_FILE)

print("Extracting exact future targets (h=1,2,4,8) from lightweight index columns...")
targets_full = pd.read_parquet(MASTER_FILE, columns=["codigo_ibge", "anio", "semana", "p_inc100k"])
targets_full = targets_full.sort_values(["codigo_ibge", "anio", "semana"]).reset_index(drop=True)

HORIZONS = [1, 2, 4, 8]
for h in HORIZONS:
    targets_full[f"target_h{h}"] = targets_full.groupby("codigo_ibge")["p_inc100k"].shift(-(h - 1))

targets_full = targets_full.drop(columns=["p_inc100k"])
bench = bench.merge(targets_full, on=["codigo_ibge", "anio", "semana"], how="left")
del targets_full

bench["persist_base"] = (bench["casos_lag_1"] / (bench["pop"] + 1e-6)) * 100000.0
bench = bench.dropna(subset=[f"target_h{h}" for h in HORIZONS] + ["persist_base"]).sort_values(["anio", "semana", "codigo_ibge"]).reset_index(drop=True)
print(f"Benchmark ready with multi-horizon targets: {bench.shape}")

exclude_cols = {
    "codigo_ibge", "municipio", "estado", "data_iniSE", "SE", "anio", "semana",
    "p_inc100k", "casos_est", "casos_est_min", "casos_est_max", "casprov",
    "nivel_inc", "nivel", "casos", "Rt", "p_rt1", "transmissao", "receptivo",
    "notif_accum_year", "persist_base", "dominant_landcover"
} | {f"target_h{h}" for h in HORIZONS}

all_num_features = [c for c in bench.select_dtypes(include=np.number).columns if c not in exclude_cols]

epi_cols = [c for c in all_num_features if c.startswith(("casos_lag", "Rt_lag", "casos_roll")) or c == "mes"]
meteo_cols = [c for c in all_num_features if any(k in c for k in ["temp", "umid", "precip", "dias_lluvia"])]
modis_cols = [c for c in all_num_features if any(k in c for k in ["NDVI", "EVI", "LST"])]
enso_cols = [c for c in all_num_features if c in ["nino34", "soi"]]
static_cols = [c for c in all_num_features if c not in (epi_cols + meteo_cols + modis_cols + enso_cols)]

class SimpleDeepForestRegressor:
    def __init__(self, n_layers=2, n_est=20, random_state=42):
        self.n_layers, self.n_est, self.random_state = n_layers, n_est, random_state
    def fit(self, X, y):
        Xc = np.asarray(X, dtype=np.float32)
        self.layers = []
        for l in range(self.n_layers):
            rf = RandomForestRegressor(n_estimators=self.n_est, max_depth=10, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=self.random_state + l)
            et = ExtraTreesRegressor(n_estimators=self.n_est, max_depth=10, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=self.random_state + 100 + l)
            rf.fit(Xc, y); et.fit(Xc, y)
            Xc = np.hstack([Xc, rf.predict(Xc)[:, None], et.predict(Xc)[:, None]]).astype(np.float32)
            self.layers.append((rf, et))
        return self
    def predict(self, X):
        Xc = np.asarray(X, dtype=np.float32)
        for rf, et in self.layers:
            rp, ep = rf.predict(Xc)[:, None], et.predict(Xc)[:, None]
            Xc = np.hstack([Xc, rp, ep]).astype(np.float32)
        return 0.5 * (rp.ravel() + ep.ravel())

def fit_predict_5layer_mlp(X_tr, y_tr, X_te):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    mdl = Sequential([
        Dense(512, activation="relu", input_shape=(X_tr.shape[1],)), BatchNormalization(),
        Dense(256, activation="relu"), BatchNormalization(),
        Dense(128, activation="relu"), BatchNormalization(),
        Dense(64, activation="relu"), BatchNormalization(),
        Dense(32, activation="relu"), BatchNormalization(),
        Dense(1)
    ])
    mdl.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)
    mdl.fit(X_tr, y_tr, validation_split=0.10, epochs=60, batch_size=8192, verbose=0, shuffle=True, callbacks=[es])
    return mdl.predict(X_te, batch_size=8192, verbose=0).ravel()

folds = [(2020, 2021), (2021, 2022), (2022, 2023), (2023, 2024), (2024, 2025)]
ALL_9_MODELS = ["LightGBM", "CatBoost", "NaivePersistence", "XGBoost", "CascadeForest", "MLP", "RandomForest", "Ridge", "LinearSVR"]

# ==============================================================================
# EXPERIMENT A: ALL 9 MODELS ACROSS MULTI-HORIZON FORECASTING (h = 1, 2, 4, 8)
# ==============================================================================
print("\n=== RUNNING EXPERIMENT A: ALL 9 MODELS MULTI-HORIZON (h=1,2,4,8) ===")
fold_metrics = {(h, m): [] for h in HORIZONS for m in ALL_9_MODELS}
fi_accum = {h: np.zeros(len(all_num_features), dtype=np.float64) for h in HORIZONS}

for fold_id, (tr_end, te_yr) in enumerate(folds, 1):
    t_f = time.time()
    print(f"\n--- Fold {fold_id}/5 (Train <= {tr_end}, Test = {te_yr}) ---")
    tr_mask = bench["anio"] <= tr_end
    te_mask = bench["anio"] == te_yr
    
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(bench.loc[tr_mask, all_num_features].astype(np.float32))
    X_te = imp.transform(bench.loc[te_mask, all_num_features].astype(np.float32))
    
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr)
    X_te_sc = sc.transform(X_te)
    p_naive = bench.loc[te_mask, "persist_base"].values.astype(np.float32)
    
    for h in HORIZONS:
        y_tr = bench.loc[tr_mask, f"target_h{h}"].values.astype(np.float32)
        y_te = bench.loc[te_mask, f"target_h{h}"].values.astype(np.float32)
        
        def rec(name, pred):
            fold_metrics[(h, name)].append((mean_absolute_error(y_te, pred), np.sqrt(mean_squared_error(y_te, pred)), r2_score(y_te, pred)))
            
        rec("NaivePersistence", p_naive)
        
        for m_name, mdl in [("Ridge", Ridge(alpha=1.0)), ("LinearSVR", LinearSVR(C=1.0, epsilon=0.0, loss="squared_epsilon_insensitive", random_state=SEED, max_iter=1500))]:
            mdl.fit(X_tr_sc, y_tr)
            rec(m_name, mdl.predict(X_te_sc))
            
        rec("MLP", fit_predict_5layer_mlp(X_tr_sc, y_tr, X_te_sc))
        
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, min_samples_leaf=20, max_features="sqrt", n_jobs=-1, random_state=SEED)
        rf.fit(X_tr, y_tr)
        rec("RandomForest", rf.predict(X_te))
        
        cf = SimpleDeepForestRegressor(n_layers=2, n_est=20, random_state=SEED)
        cf.fit(X_tr, y_tr)
        rec("CascadeForest", cf.predict(X_te))
        
        lgb = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, max_bin=127, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
        lgb.fit(X_tr, y_tr)
        fi_accum[h] += lgb.feature_importances_
        rec("LightGBM", lgb.predict(X_te))
        
        xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=SEED, tree_method="hist", device="cuda")
        xgb.fit(X_tr, y_tr)
        rec("XGBoost", xgb.predict(X_te))
        
        cat = CatBoostRegressor(iterations=200, learning_rate=0.05, depth=6, random_seed=SEED, task_type="GPU", verbose=0)
        cat.fit(X_tr, y_tr)
        rec("CatBoost", cat.predict(X_te))
        
    print(f"Fold {fold_id} completed in {(time.time() - t_f):.1f}s")

mh_rows, modality_share_rows = [], []
for h in HORIZONS:
    for m_name in ALL_9_MODELS:
        arr = np.array(fold_metrics[(h, m_name)])
        mh_rows.append({
            "Horizon_Weeks": h, "Model": m_name,
            "MAE_Mean": arr[:, 0].mean(), "MAE_SD": arr[:, 0].std(),
            "RMSE_Mean": arr[:, 1].mean(), "RMSE_SD": arr[:, 1].std(),
            "R2_Mean": arr[:, 2].mean(), "R2_SD": arr[:, 2].std()
        })
    fi_s = pd.Series(fi_accum[h] / len(folds), index=all_num_features)
    tot = fi_s.sum() + 1e-9
    modality_share_rows.append({
        "Horizon_Weeks": h,
        "Epidemiology_Pct": 100.0 * fi_s[epi_cols].sum() / tot,
        "Meteo_CHIRPS_Pct": 100.0 * fi_s[meteo_cols].sum() / tot,
        "MODIS_Satellite_Pct": 100.0 * fi_s[modis_cols].sum() / tot,
        "ENSO_Pct": 100.0 * fi_s[enso_cols].sum() / tot,
        "Static_Context_Pct": 100.0 * fi_s[static_cols].sum() / tot
    })

mh_df = pd.DataFrame(mh_rows)
mod_share_df = pd.DataFrame(modality_share_rows)
print("\n=== TABLE: MULTI-HORIZON PERFORMANCE (ALL 9 MODELS) ===")
print(mh_df.to_string(index=False))
print("\n=== TABLE: MODALITY IMPORTANCE SHIFT ACROSS HORIZONS ===")
print(mod_share_df.to_string(index=False))

# ==============================================================================
# EXPERIMENT B: MODALITY ABLATION STUDY (h = 1 and h = 4 weeks)
# ==============================================================================
print("\n=== RUNNING EXPERIMENT B: MODALITY ABLATION ===")
ablation_configs = [
    ("M1: Epidemiology Only", epi_cols),
    ("M2: + Meteo & CHIRPS", epi_cols + meteo_cols),
    ("M3: + MODIS Satellite", epi_cols + meteo_cols + modis_cols),
    ("M4: + ENSO Teleconnections", epi_cols + meteo_cols + modis_cols + enso_cols),
    ("M5: Full Multimodal (+ Static)", all_num_features)
]

abl_rows = []
for h in [1, 4]:
    target_col = f"target_h{h}"
    for cfg_name, feat_subset in ablation_configs:
        f_scores = []
        for tr_end, te_yr in folds:
            tr_mask, te_mask = bench["anio"] <= tr_end, bench["anio"] == te_yr
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(bench.loc[tr_mask, feat_subset].astype(np.float32))
            X_te = imp.transform(bench.loc[te_mask, feat_subset].astype(np.float32))
            y_tr, y_te = bench.loc[tr_mask, target_col].values.astype(np.float32), bench.loc[te_mask, target_col].values.astype(np.float32)
            lgb = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, max_bin=127, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
            lgb.fit(X_tr, y_tr)
            p = lgb.predict(X_te)
            f_scores.append((mean_absolute_error(y_te, p), np.sqrt(mean_squared_error(y_te, p)), r2_score(y_te, p)))
        arr = np.array(f_scores)
        abl_rows.append({
            "Horizon_Weeks": h, "Configuration": cfg_name, "Num_Features": len(feat_subset),
            "MAE_Mean": arr[:, 0].mean(), "RMSE_Mean": arr[:, 1].mean(), "R2_Mean": arr[:, 2].mean(), "R2_SD": arr[:, 2].std()
        })

abl_df = pd.DataFrame(abl_rows)
print("\n=== TABLE: MODALITY ABLATION ===")
print(abl_df.to_string(index=False))

# ==============================================================================
# EXPERIMENT C: SPATIAL GENERALIZATION (LEAVE-MACRO-REGION-OUT, h = 1 & h = 4)
# ==============================================================================
print("\n=== RUNNING EXPERIMENT C: SPATIAL TRANSFER (LEAVE-MACRO-REGION-OUT) ===")
region_map = {"1": "North (Amazon)", "2": "Northeast", "3": "Southeast", "4": "South", "5": "Central-West"}
bench["macro_region"] = bench["codigo_ibge"].astype(str).str.strip().str[0].map(region_map)

sp_rows = []
for h in [1, 4]:
    target_col = f"target_h{h}"
    for reg_code, reg_name in region_map.items():
        tr_out = (bench["macro_region"] != reg_name) & (bench["anio"] <= 2023)
        te_reg = (bench["macro_region"] == reg_name) & (bench["anio"] >= 2024)
        
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(bench.loc[tr_out, all_num_features].astype(np.float32))
        X_te = imp.transform(bench.loc[te_reg, all_num_features].astype(np.float32))
        y_tr = bench.loc[tr_out, target_col].values.astype(np.float32)
        y_te = bench.loc[te_reg, target_col].values.astype(np.float32)
        
        lgb = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, max_bin=127, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
        lgb.fit(X_tr, y_tr)
        preds_out = lgb.predict(X_te)
        
        tr_in = (bench["macro_region"] == reg_name) & (bench["anio"] <= 2023)
        imp_in = SimpleImputer(strategy="median")
        X_tr_in = imp_in.fit_transform(bench.loc[tr_in, all_num_features].astype(np.float32))
        X_te_in = imp_in.transform(bench.loc[te_reg, all_num_features].astype(np.float32))
        lgb_in = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, max_bin=127, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbose=-1)
        lgb_in.fit(X_tr_in, bench.loc[tr_in, target_col].values.astype(np.float32))
        preds_in = lgb_in.predict(X_te_in)
        
        r2_out, r2_in = r2_score(y_te, preds_out), r2_score(y_te, preds_in)
        sp_rows.append({
            "Horizon_Weeks": h, "Target_Region": reg_name, "Test_Samples": int(te_reg.sum()),
            "Within_Region_R2": r2_in, "ZeroShot_Transfer_R2": r2_out, "Spatial_Transfer_Gap": r2_in - r2_out,
            "ZeroShot_MAE": mean_absolute_error(y_te, preds_out)
        })

sp_df = pd.DataFrame(sp_rows)
print("\n=== TABLE: SPATIAL TRANSFERABILITY ===")
print(sp_df.to_string(index=False))

mh_df.to_csv(os.path.join(OUT_DIR, "multi_horizon_9models.csv"), index=False)
mod_share_df.to_csv(os.path.join(OUT_DIR, "modality_share_horizon.csv"), index=False)
abl_df.to_csv(os.path.join(OUT_DIR, "modality_ablation.csv"), index=False)
sp_df.to_csv(os.path.join(OUT_DIR, "spatial_transfer.csv"), index=False)

with pd.ExcelWriter(os.path.join(OUT_DIR, "extended_scientific_results.xlsx")) as writer:
    mh_df.to_excel(writer, sheet_name="MULTI_HORIZON_9MODELS", index=False)
    mod_share_df.to_excel(writer, sheet_name="MODALITY_SHARE_HORIZON", index=False)
    abl_df.to_excel(writer, sheet_name="MODALITY_ABLATION", index=False)
    sp_df.to_excel(writer, sheet_name="SPATIAL_TRANSFER", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4), dpi=300)
plot_styles = [
    ("LightGBM", "o", "-"), ("CatBoost", "s", "-"), ("XGBoost", "^", "-"),
    ("CascadeForest", "v", "-"), ("MLP", "P", "-"), ("RandomForest", "D", "-"),
    ("NaivePersistence", "X", "--"), ("Ridge", "d", ":")
]
for m_name, marker, ls in plot_styles:
    sub = mh_df[mh_df["Model"] == m_name]
    axes[0].plot(sub["Horizon_Weeks"], sub["R2_Mean"], marker=marker, linestyle=ls, lw=1.8, label=m_name)
axes[0].set_xlabel("Forecasting Horizon $h$ (Weeks Ahead)", fontweight="bold")
axes[0].set_ylabel("Mean Coefficient of Determination ($R^2$)", fontweight="bold")
axes[0].set_title("(a) Prospective Forecasting Decay Across Horizons", fontweight="bold")
axes[0].set_xticks(HORIZONS)
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=8.2, ncol=2)

mod_cols_plot = ["Epidemiology_Pct", "Meteo_CHIRPS_Pct", "MODIS_Satellite_Pct", "ENSO_Pct", "Static_Context_Pct"]
mod_labels = ["Epidemiology", "Meteo & CHIRPS", "MODIS Satellite", "ENSO Indices", "Static Context"]
bottom = np.zeros(len(HORIZONS))
colors = ["#1e293b", "#0284c7", "#16a34a", "#d97706", "#7c3aed"]
for col, lbl, c in zip(mod_cols_plot, mod_labels, colors):
    vals = mod_share_df[col].values
    axes[1].bar([str(h) for h in HORIZONS], vals, bottom=bottom, label=lbl, color=c, width=0.55)
    bottom += vals
axes[1].set_xlabel("Forecasting Horizon $h$ (Weeks Ahead)", fontweight="bold")
axes[1].set_ylabel("Relative Split Importance Share (%)", fontweight="bold")
axes[1].set_title("(b) Modality Contribution Shift Across Horizons", fontweight="bold")
axes[1].legend(loc="upper right", fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Figure_7_MultiHorizon_ModalityShift.png"), dpi=300)
plt.close()

print(f"\n[✓] Completed all 9-model multi-horizon, ablation & spatial transfer experiments in {(time.time() - start_time)/60:.2f} min.")