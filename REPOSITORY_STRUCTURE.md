# Repository Structure

This repository contains the data processing pipeline, benchmark experiments, documentation and manuscript materials associated with:

**DATASET_MULTIMODAL_V8: A Nationwide Multimodal Dataset for Dengue Surveillance, Nowcasting and Forecasting in Brazil (2010–2025)**

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
│   ├── 28_build_benchmark_dataset.py
│   ├── 29_benchmark_lightgbm.py
│   ├── 30_benchmark_xgboost.py
│   ├── 31_benchmark_catboost.py
│   ├── 32_benchmark_random_forest.py
│   └── 33_compare_models.py
│
├── benchmarks/
│   ├── LIGHTGBM/
│   ├── LIGHTGBM_V3_FORECASTING/
│   ├── XGBOOST_V3_FORECASTING/
│   ├── CATBOOST_V3_FORECASTING/
│   ├── RANDOM_FOREST_V3_FORECASTING/
│   └── COMPARISON/
│
├── README.md
├── LICENSE
├── CITATION.cff
└── REPOSITORY_STRUCTURE.md
└── CHANGELOG.md
└── DATASET_CARD.md


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

dataset_multimodal_v8.parquet

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

Contains all processing pipelines used to generate the final dataset.

Scripts are numbered according to execution order.

Examples:

### Block 0

Geographic base generation.

### Block 1

Epidemiological data integration.

### Block 2

Precipitation data integration.

### Block 3

MODIS feature extraction.

### ...

### Block 10

Temporal feature engineering.

---

## benchmarks/

Machine learning benchmark experiments.

Includes:

- LightGBM
- XGBoost
- CatBoost
- Random Forest

and model comparison outputs.

---

## paper/

Materials related to the accompanying scientific publication.

Includes:

- manuscript drafts
- tables
- figures
- supplementary material

---

# Reproducibility

The dataset can be fully reproduced by executing the scripts in numerical order.

All benchmark experiments were conducted using the benchmark dataset derived from:

dataset_multimodal_v8.parquet

and temporal expanding-window validation.

---

# Contact

J. Salvador Sánchez (sanchez@uji.es)

Project:

Early Detection of Dengue Outbreaks in Brazil
