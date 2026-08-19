# DATASET CARD

## Dataset Name

DATASET_MULTIMODAL_V8

## Version

v8

## Status

Stable baseline release.

## Dataset Summary

DATASET_MULTIMODAL_V8 is a nationwide municipal-level multimodal dataset designed for dengue surveillance, nowcasting, forecasting, epidemiological analysis and machine learning benchmarking in Brazil.

The dataset integrates epidemiological, meteorological, precipitation, remote sensing, climate teleconnection, topographic, land-cover, demographic and sanitation information into a unified weekly municipal database covering the period 2010–2025.

---

## Coverage

| Property | Value |
|-----------|---------|
| Country | Brazil |
| Municipalities | 5,570 |
| States | 27 |
| Records | 4,701,298 |
| Variables | 87 |
| Temporal Coverage | 2010–2025 |
| Temporal Resolution | Weekly |
| Spatial Resolution | Municipality |
| Dataset Version | v8 |

---

## Scientific Motivation

Dengue transmission is driven by complex interactions among epidemiological, climatic, environmental, demographic and infrastructural factors.

DATASET_MULTIMODAL_V8 was developed to integrate these heterogeneous information sources into a single nationwide database suitable for:

- Dengue surveillance
- Outbreak prediction
- Machine learning benchmarks
- Explainable AI (XAI)
- Spatial epidemiology
- Climate-health studies
- Deep learning applications
- Epidemiology-informed AI systems

---

## Data Sources

### Epidemiology

Source:

- InfoDengue

Variables include:

- casos
- casos_est
- casos_est_min
- casos_est_max
- casprov
- p_inc100k
- Rt
- p_rt1
- nivel
- nivel_inc
- transmissao
- receptivo

---

### Meteorology

Variables:

- tempmin
- tempmed
- tempmax
- umidmin
- umidmed
- umidmax

---

### Precipitation

Source:

- CHIRPS v2.0

Variables:

- precip_total_semana
- precip_media_semana
- precip_max_semana
- dias_lluvia_semana

---

### Remote Sensing

Source:

- MODIS Vegetation Indices
- MODIS Land Surface Temperature

Variables:

- NDVI_mean
- NDVI_std
- EVI_mean
- EVI_std
- LST_Day_mean
- LST_Day_std
- LST_Night_mean
- LST_Night_std

---

### Climate Teleconnections

Source:

- NOAA Climate Prediction Center

Variables:

- nino34
- soi

---

### Topography

Source:

- Copernicus DEM GLO-30

Variables:

- elev_min
- elev_mean
- elev_max

---

### Land Cover

Source:

- ESA WorldCover 10m

Variables:

- pct_tree_cover
- pct_shrubland
- pct_grassland
- pct_cropland
- pct_builtup
- pct_water
- pct_wetland
- pct_bare_sparse
- pct_mangroves
- dominant_landcover

---

### Demography

Source:

- IBGE Census 2022

Variables:

- population_total
- population_density
- male_population_pct
- female_population_pct
- median_age
- aging_index
- sex_ratio

---

### Sanitation

Source:

- Instituto Água e Saneamento

Variables:

- water_supply_pct
- sewage_pct
- garbage_collection_pct

---

## Feature Engineering

### Lag Variables

The dataset includes 13 lag variables:

- casos_lag_1
- casos_lag_2
- casos_lag_4
- casos_lag_8
- Rt_lag_1
- Rt_lag_2
- Rt_lag_4
- precip_total_semana_lag_1
- precip_total_semana_lag_4
- tempmed_lag_1
- tempmed_lag_4
- umidmed_lag_1
- umidmed_lag_4

### Rolling Window Features

The dataset includes 6 rolling-window variables:

- casos_roll4_mean
- casos_roll8_mean
- precip_roll4_sum
- precip_roll8_sum
- tempmed_roll4_mean
- tempmed_roll8_mean

---

## Reproducibility

The dataset can be fully reproduced using the processing pipeline included in this repository.

The current release was generated through a sequence of data integration, feature engineering and benchmarking scripts covering:

- Epidemiology
- Meteorology
- Precipitation
- Remote sensing
- Climate teleconnections (ENSO)
- Topography
- Land cover
- Demography
- Sanitation
- Temporal lag generation
- Rolling-window feature construction

The complete pipeline consists of 34 scripts executed sequentially.

Benchmark experiments reported in this dataset card were conducted using temporal expanding-window validation and a benchmark subset derived from DATASET_MULTIMODAL_V8.

The repository includes all scripts required to reproduce both the dataset and the benchmark experiments.

---

## Data Quality Assessment

### Administrative Consistency

- Municipal coverage: 5,570 / 5,570 municipalities
- State coverage: 27 states
- Municipality matching rate: 100%
- Duplicate municipality identifiers: 0

### Structural Audit

The dataset successfully passed the structural audit.

Validation checks included:

- Geographic consistency
- Administrative consistency
- Temporal consistency
- Duplicate detection
- Municipality matching

### Duplicate Rows

- Duplicate records detected: 0

---

## Missing Values

Most variables present complete or near-complete coverage.

Main sources of missing observations include:

### Sanitation

| Variable | Missing (%) |
|-----------|------------:|
| sewage_pct | 51.13 |
| garbage_collection_pct | 12.92 |
| water_supply_pct | 8.76 |

### Meteorology

| Variable | Missing (%) |
|-----------|------------:|
| umidmed | 8.59 |
| umidmin | 8.27 |
| umidmax | 7.12 |
| tempmed | 8.56 |
| tempmax | 8.56 |
| tempmin | 4.37 |

### MODIS

| Variable | Missing (%) |
|-----------|------------:|
| LST_Night_mean | 9.21 |
| LST_Night_std | 9.21 |
| LST_Day_mean | 4.56 |
| LST_Day_std | 4.56 |

Notes:

- Missing values originate from source availability and satellite quality filters.
- Missing values are not caused by the integration process.
- Municipality matching coverage is complete.

---

## Benchmark Results

### Nowcasting Scenario

LightGBM achieved:

| Model | R² |
|---------|---------:|
| LightGBM | 0.9869 |

### Forecasting Scenario

| Model | R² |
|---------|---------:|
| LightGBM | 0.7843 |
| CatBoost | 0.7820 |
| XGBoost | 0.7675 |
| Random Forest | 0.6447 |

### Nowcasting vs Forecasting

| Scenario | Model | R² |
|-----------|-----------|----------:|
| Nowcasting | LightGBM | 0.9869 |
| Forecasting | LightGBM | 0.7843 |

Difference:

ΔR² = -0.2026

### Statistical Comparison

Friedman Test:

- p-value = 0.004996

The forecasting benchmark identified statistically significant differences among the evaluated model families.

---

## Intended Uses

The dataset is intended for:

- Dengue forecasting
- Dengue outbreak prediction
- Machine learning benchmarking
- Environmental epidemiology
- Climate-health studies
- Explainable AI
- Spatial epidemiology
- Deep learning applications
- Epidemiology-informed AI systems

---

## Known Limitations

- Missing sanitation observations remain in several municipalities.
- Human mobility information is not included.
- Healthcare infrastructure indicators are not included.
- Socioeconomic indicators such as HDI and income per capita are not included.
- The dataset currently covers Brazil only.

---

## Ethical Considerations

The dataset contains municipality-level aggregated information.

No personally identifiable information (PII) is included.

No individual-level health records are provided.

---

## Version History

### v1

Epidemiology + precipitation + MODIS

### v2

v1 + ENSO

### v3

v2 + elevation

### v4

v3 + land cover

### v5

v4 + demography

### v6

v5 + area_km2 + population_density

### v7

v6 + sanitation variables

### v8

v7 + lag variables + rolling-window features

---

## Availability

### GitHub Repository

https://github.com/jssanchez-garreta/dataset-multimodal-dengue-brazil

### Zenodo Archive

DOI: 10.5281/zenodo.XXXXXXXX

URL:
https://doi.org/10.5281/zenodo.XXXXXXXX

### Citation

Please cite this dataset using the information provided in:

- CITATION.cff

---

## Authors

- Raquel Sánchez-Marqués (MRC Unit The Gambia at LSHTM)
- J. Salvador Sánchez (Universitat Jaume I)

## Contact

Project:

Early Detection of Dengue Outbreaks in Brazil

Corresponding author:

J. Salvador Sánchez

Email:

sanchez@uji.es

---

## License

CC BY 4.0

## Recommended Citation

See CITATION.cff.
