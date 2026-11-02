# Repository Structure

This repository contains the data processing pipeline, benchmark experiments, documentation and manuscript materials associated with:

**BraDengue-MM: A Nationwide Multimodal Dataset for Dengue Surveillance, Nowcasting and Forecasting in Brazil (2010–2025)**

---

# Directory Tree
.
├── data/
│   ├── raw/
│   ├── intermediate/
│   └── processed/
│
├── metadata/
│   ├── DATASET_CARD.md
│   ├── DATA_DICTIONARY_V8.csv
│   ├── AUDIT_V8.md
│   ├── EDA_V8_REPORT.md
│   └── dataset_multimodal_metadata.md
│
├── scripts/
│   ├── 00_build_municipality_master.py
│   ├── ...
│   ├── 26_eda_v8.py
│   ├── 28_build_benchmark_dataset.py
│   ├── 29_benchmark_lightgbm.py
│   ├── 30_benchmark_xgboost.py
│   ├── 31_benchmark_catboost.py
│   ├── 32_benchmark_random_forest.py
│   ├── 33_benchmark_deep_forest_simplified.py
│   ├── 34_benchmark_ridge.py
│   ├── 35_benchmark_naive_persistence.py
│   ├── 36_benchmark_svr.py
│   ├── 37_benchmark_mlp_layers.py
│   └── 40_compare_models.py
│
├── benchmarks/
│   ├── LIGHTGBM/
│   ├── XGBOOST/
│   ├── CATBOOST/
│   ├── RANDOM_FOREST/
│   ├── DEEP_FOREST/
│   ├── RIDGE/
│   ├── NAIVE_PERSISTENCE/
│   ├── SVR/
│   ├── MLP/
│   └── COMPARISON/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── REPOSITORY_STRUCTURE.md
└── CHANGELOG.md


---

# Main Components

## data/

Contains dataset files.

### raw/

Original source datasets before processing.

### intermediate/

Intermediate datasets generated during integration.

### processed/

Final datasets used in experiments.

Main file:

`BraDengue-MM.parquet`

---

## metadata/

Contains documentation and dataset description.

### DATASET_CARD.md

High-level description of the dataset.

### DATA_DICTIONARY_V8.csv

Complete variable dictionary.

### AUDIT_V8.md

Automated structural audit.

### EDA_V8_REPORT.md

Exploratory data analysis report.

### dataset_multimodal_metadata.md

Full dataset documentation and construction process.

---

## scripts/

Contains all processing pipelines used to generate the final dataset and run benchmark experiments.

Scripts are numbered according to execution order.

### Data Processing Pipeline (00 – 25)

- **Block 0:** Geographic base generation (`00_build_municipality_master.py`)
- **Block 1:** Epidemiological data integration
- **Block 2:** Precipitation data integration
- **Block 3:** MODIS feature extraction
- **Block 4–9:** ENSO, elevation, land cover, demography, population density, and sanitation integration
- **Block 10:** Temporal feature engineering (`24_build_temporal_lags.py`)

### Analysis and Dataset Preparation (26 – 28)

- `26_eda_v8.py`: Exploratory data analysis for release v8.
- `27_generate_paper_tables.py`: Script to generate summary tables for the manuscript.
- `28_build_benchmark_dataset.py`: Derived benchmark dataset sampling and filtering pipeline.

### Machine Learning Benchmarking (29 – 40)

- `29_benchmark_lightgbm.py`: LightGBM gradient boosting benchmark.
- `30_benchmark_xgboost.py`: XGBoost gradient boosting benchmark.
- `31_benchmark_catboost.py`: CatBoost gradient boosting benchmark.
- `32_benchmark_random_forest.py`: Random Forest ensemble benchmark.
- `33_benchmark_deep_forest.py` & `33_benchmark_deep_forest_simplified.py`: Cascade Forest / Deep Forest decision cascade benchmarks.
- `34_benchmark_ridge.py`: Ridge regularized linear regression baseline.
- `35_benchmark_naive_persistence.py`: Naïve persistence baseline ($y_{t} = y_{t-1}$).
- `36_benchmark_svr.py`: Support Vector Regression (LinearSVR) baseline.
- `37_benchmark_mlp_layers.py` & `38_benchmark_mlp_v2.py`: Multi-Layer Perceptron (Deep Feedforward NN) architecture evaluation.
- `40_compare_models.py`: Statistical aggregation and comparison across all model benchmarks.

---

## benchmarks/

Machine learning benchmark experiments outputs and metrics across 5 predictive paradigms:

- **Tree Ensembles & Gradient Boosting:** LightGBM, CatBoost, XGBoost, Random Forest
- **Deep Decision Cascades:** Cascade Forest
- **Deep Neural Networks:** 5-Layer MLP
- **Regularized Linear & SVM:** Ridge Regression, LinearSVR
- **Naïve Baseline:** Persistence Model
- **Model Comparison:** Summary statistical evaluations and plots

---

# Reproducibility

The dataset and experiment results can be fully reproduced by executing the scripts in numerical order from `00` to `40`.

All benchmark experiments were conducted using the benchmark dataset derived from:

`BraDengue-MM.parquet`

and evaluated under temporal expanding-window validation (2021–2025).

---

# Contact

Corresponding author: **J. Salvador Sánchez** (sanchez@uji.es)

Project:

Early Detection of Dengue Outbreaks in Brazil
