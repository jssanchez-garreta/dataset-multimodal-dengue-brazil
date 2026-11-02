# BraDengue-MM

## A Nationwide Multimodal Dataset for Dengue Surveillance, Nowcasting and Forecasting in Brazil (2010–2025)

### Overview

BraDengue-MM is a nationwide municipal-level multimodal dataset designed for dengue surveillance, epidemiological analysis, outbreak prediction and machine learning applications in Brazil.

The dataset integrates epidemiological, meteorological, precipitation, remote sensing, climate teleconnection, topographic, land-cover, demographic and sanitation information into a unified weekly municipal database covering 2010–2025.

### Availability

#### Source Code, Documentation and Benchmarks
- **GitHub Repository:** https://github.com/jssanchez-garreta/dataset-multimodal-dengue-brazil
- **Zenodo Repository DOI:** https://doi.org/10.5281/zenodo.22027186

#### Datasets
- **Zenodo Dataset DOI:** https://doi.org/10.5281/zenodo.22029052
- **Included Files:**
  - `BraDengue-MM.parquet`
  - `dataset_benchmark_v1.parquet`

### Key Characteristics

| Property | Value |
|-----------|---------|
| Country | Brazil |
| Municipalities | 5,570 |
| States | 27 |
| Records | 4,634,540 |
| Variables | 87 |
| Temporal Coverage | 2010–2025 |
| Temporal Resolution | Weekly |
| Spatial Resolution | Municipality |

### Benchmark Results

#### Prospective Forecasting (Expanding-Window Validation 2021–2025)

| Model | MAE | RMSE | R² | Rank |
|:---|:---:|:---:|:---:|:---:|
| **LightGBM** | 16.49 | 57.31 | **0.8010** | 1 |
| **CatBoost** | 16.63 | 57.53 | **0.7999** | 2 |
| **Naïve Persistence** | 16.61 | 56.80 | **0.7974** | 3 |
| **XGBoost** | 17.07 | 61.30 | **0.7784** | 4 |
| **Cascade Forest** | 18.08 | 65.71 | **0.7482** | 5 |
| **5-Layer MLP** | 22.96 | 74.70 | **0.6831** | 6 |
| **Random Forest** | 20.40 | 79.21 | **0.6535** | 7 |
| **Ridge Regression** | 39.85 | 132.98 | **0.0483** | 8 |
| **LinearSVR** | 39.85 | 132.98 | **0.0483** | 9 |

#### Nowcasting vs. Forecasting Gap

| Scenario | Model | R² |
|:---|:---|:---:|
| **Nowcasting** | LightGBM | **0.9869** |
| **Forecasting** | LightGBM | **0.8010** |
| **Optimism Bias Gap** | — | **ΔR² = 0.1860** |

## Authors

- **J. Salvador Sánchez** (Universitat Jaume I)
- **Raquel Sánchez-Marqués** (MRC Unit The Gambia at LSHTM)

## Contact

Corresponding author: **J. Salvador Sánchez** (sanchez@uji.es)

## License

CC BY 4.0

### Citation

If you use BraDengue-MM in scientific work, please cite the dataset using the information provided in `CITATION.cff`.
