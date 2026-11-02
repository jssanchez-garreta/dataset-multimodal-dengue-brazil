# Changelog

## v8

Current stable baseline release.

### New Additions
- Demography and Population Density layers (IBGE Census 2022).
- Sanitation indicators (Instituto Água e Saneamento).
- 13 Lag variables and 6 Rolling-window features.
- Unified nomenclature established: **BraDengue-MM**.

### Benchmark Results

#### Nowcasting
- LightGBM: R² = 0.9869

#### Prospective Forecasting (5-Fold Temporal Validation)
- LightGBM: R² = 0.8010 (MAE = 16.49)
- CatBoost: R² = 0.7999
- Naïve Persistence: R² = 0.7974
- XGBoost: R² = 0.7784
- Cascade Forest: R² = 0.7482
- 5-Layer MLP: R² = 0.6831
- Random Forest: R² = 0.6535
- Ridge Regression / LinearSVR: R² = 0.0483

#### Methodological Findings
- Quantified optimism bias / performance gap between nowcasting and leakage-free forecasting: ΔR² = 0.1860.

### Repository Publication
- GitHub repository published: https://github.com/jssanchez-garreta/dataset-multimodal-dengue-brazil
- Zenodo DOI assigned: 10.5281/zenodo.22027186