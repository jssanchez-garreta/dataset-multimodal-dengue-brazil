# DATASET_MULTIMODAL_V8

## A Nationwide Multimodal Dataset for Dengue Surveillance, Nowcasting and Forecasting in Brazil (2010–2025)

### Overview

DATASET_MULTIMODAL_V8 is a nationwide municipal-level multimodal dataset designed for dengue surveillance, epidemiological analysis, outbreak prediction and machine learning applications in Brazil.

The dataset integrates epidemiological, meteorological, precipitation, remote sensing, climate teleconnection, topographic, land-cover, demographic and sanitation information into a unified weekly municipal database covering 2010–2025.

## Availability

The dataset is publicly available through:

- GitHub repository
- Zenodo archive (DOI pending)

The current stable release is DATASET_MULTIMODAL_V8.

### Key Characteristics

| Property | Value |
|-----------|---------|
| Records | 4,701,298 |
| Variables | 87 |
| Municipalities | 5,570 |
| States | 27 |
| Temporal Coverage | 2010–2025 |
| Temporal Resolution | Weekly |
| Spatial Resolution | Municipality |

### Modalities

- Epidemiology
- Meteorology
- Precipitation
- MODIS Remote Sensing
- ENSO Climate Indices
- Topography
- Land Cover
- Demography
- Sanitation
- Temporal Lag Features
- Rolling Window Features

### Benchmark Results

#### Nowcasting

| Model | R² |
|---------|---------:|
| LightGBM | 0.9869 |

#### Forecasting

| Model | R² |
|---------|---------:|
| LightGBM | 0.7843 |
| CatBoost | 0.7820 |
| XGBoost | 0.7675 |
| Random Forest | 0.6447 |

### Repository Structure

data/
metadata/
benchmarks/
scripts/

### Documentation

Detailed documentation is available in:

- DATASET_CARD.md — Complete dataset description and usage guidance.
- REPOSITORY_STRUCTURE.md — Repository organization and directory structure.
- DATA_DICTIONARY_V8.csv — Variable dictionary and metadata.
- AUDIT_V8.md — Automated structural audit report.
- EDA_V8_REPORT.md — Exploratory data analysis report.
- dataset_multimodal_metadata.md — Full dataset construction and integration process.

## Authors

- Raquel Sánchez-Marqués
- J. Salvador Sánchez

## Contact

Corresponding author:

José Salvador Sánchez Garreta  
sanchez@uji.es

## License

CC BY 4.0

### Citation

If you use DATASET_MULTIMODAL_V8 in scientific work, please cite the dataset using the information provided in CITATION.cff.