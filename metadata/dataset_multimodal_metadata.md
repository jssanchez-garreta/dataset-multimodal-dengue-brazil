# DATASET_MULTIMODAL_V8

## Overview

DATASET_MULTIMODAL_V8 is a municipal-level multimodal dataset designed for dengue surveillance, epidemiological analysis, outbreak prediction and machine learning applications in Brazil.

The dataset integrates epidemiological, meteorological, precipitation, remote sensing, climate teleconnection, topographic, land-cover, demographic and sanitation information into a unified weekly municipal database covering the period 2010–2025.

The dataset was constructed through a multi-stage integration pipeline combining official Brazilian data sources, satellite-derived environmental indicators and global climate datasets.

Current status:

- Version: v8
- Stable baseline release
- Ready for exploratory analysis and benchmark modeling

---

# Dataset Dimensions

| Property | Value |
|-----------|---------|
| Records | 4,701,298 |
| Variables | 87 |
| Municipalities | 5,570 |
| States | 27 |
| Temporal Coverage | 2010–2025 |
| Temporal Resolution | Weekly |
| Spatial Resolution | Municipality |
| Country | Brazil |
| Dataset Version | v8 |

---

## Executive Summary

DATASET_MULTIMODAL_V8 integrates:

- Epidemiological data from InfoDengue
- Local meteorology
- CHIRPS precipitation
- MODIS vegetation indices
- MODIS land surface temperature
- ENSO indicators
- Elevation
- Land cover
- Demography
- Population density
- Sanitation
- Temporal lag variables and rolling windows

The final dataset contains 4,701,298 weekly observations covering all 5,570 Brazilian municipalities during the period 2010-2025.

---

## Main Strengths

- Nationwide municipal coverage.
- Weekly temporal resolution.
- 16-year time span (2010-2025).
- Integration of epidemiological, environmental and socioeconomic information.
- Full municipality matching (5570 / 5570).
- MODIS vegetation coverage of 100%.
- Elevation coverage of 100%.
- Land-cover coverage of 100%.

---

## Scientific Motivation

Dengue transmission is driven by complex interactions between epidemiological, climatic, environmental, demographic and infrastructural factors.

DATASET_MULTIMODAL_V8 was designed to integrate these heterogeneous information sources into a unified nationwide database suitable for epidemiological studies, machine learning modeling, explainable artificial intelligence and outbreak prediction.

The dataset aims to facilitate the development, benchmarking and interpretation of predictive models operating at municipal scale across Brazil.

---

# Geographic Reference Dataset

The project is based on the official municipality boundaries provided by IBGE.

Source:

https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais/15774-malhas.html

Reference files:

- municipios_brasil_master.csv
- municipios_brasil_master.gpkg

The master geographic dataset provides:

- Municipal code (IBGE)
- Municipality name
- State
- Municipality area
- Municipality centroid
- Municipality geometry

The GeoPackage version is used for:

- GIS analysis
- Spatial visualization
- Spatial graph construction
- Land-cover integration
- MODIS integration
- Spatial epidemiological analysis

---

# Dataset Construction Pipeline

The final dataset was generated through 10 sequential integration stages.

## Block 0 – Geographic Base

Source:

IBGE Municipal Boundaries 2025

Outputs:

- municipios_brasil_master.csv
- municipios_brasil_master.gpkg

---

## Block 1 – Epidemiological Data

Source:

InfoDengue

https://info.dengue.mat.br

Outputs:

- epidemiologia_infodengue_2010_2025_clean.csv
- epidemiologia_infodengue_2010_2025_clean.parquet

Coverage:

5570 of 5573 geographic entities.

Missing entities:

- Boa Esperança do Norte (MT)
- Área Operacional Lagoa Mirim (RS)
- Área Operacional Lagoa dos Patos (RS)

---

## Block 2 – Precipitation

Source:

CHIRPS v2.0

Variables:

- precip_total_semana
- precip_media_semana
- precip_max_semana
- dias_lluvia_semana

Outputs:

- precipitacion_chirps_2010_2025.parquet

---

## Block 3 – Remote Sensing

Source:

Microsoft Planetary Computer

Datasets:

- MODIS Vegetation Indices 16-Day (250m)
- MODIS Land Surface Temperature 8-Day

Variables:

- NDVI_mean
- NDVI_std
- EVI_mean
- EVI_std
- LST_Day_mean
- LST_Day_std
- LST_Night_mean
- LST_Night_std

Coverage:

- NDVI: 100%
- EVI: 100%
- LST Day: 97.99%
- LST Night: 93.19%

Outputs:

- modis_brasil_2010_2025_master.parquet

---

## Block 4 – ENSO

Source:

NOAA Climate Prediction Center

Variables:

- nino34
- soi

Outputs:

- enso_2010_2025.parquet

---

## Block 5 – Elevation

Source:

Copernicus DEM GLO-30

Resolution:

30 meters

Coverage:

100%

Variables:

- elev_mean
- elev_min
- elev_max
- n_pixels
- n_tiles

Outputs:

- elevation_brasil.parquet

---

## Block 6 – Land Cover

Source:

ESA WorldCover 10m

Variables:

- pct_tree_cover
- pct_shrubland
- pct_grassland
- pct_cropland
- pct_builtup
- pct_bare_sparse
- pct_water
- pct_wetland
- pct_mangroves
- dominant_landcover

Outputs:

- landcover_brasil.parquet

---

## Block 7 – Demography

Source:

IBGE Census 2022 (SIDRA)

Variables:

- population_total
- male_population_pct
- female_population_pct
- aging_index
- median_age
- sex_ratio

Outputs:

- demographics_brasil.parquet

---

## Block 8 – Population Density

Derived variables:

- area_km2
- population_density

Outputs:

- population_density_brasil.parquet

---

## Block 9 – Sanitation

Source: Instituto Água e Saneamento

https://www.aguaesaneamento.org.br

Variables:

- water_supply_pct
- sewage_pct
- garbage_collection_pct

Outputs:

- sanitation_brasil.parquet

Municipality matching: 5570 / 5570 municipalities

Matching rate: 100%

Duplicate municipality IDs: 0

---

## Block 10 – Temporal Lag Variables and Rolling Windows

Generated variables:

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
- casos_roll4_mean
- casos_roll8_mean
- precip_roll4_sum
- precip_roll8_sum
- tempmed_roll4_mean
- tempmed_roll8_mean

---

# Variable Inventory

## Administrative Variables

- codigo_ibge
- municipio
- estado

## Temporal Variables

- data_iniSE
- SE
- anio
- semana
- mes

## Epidemiological Variables

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
- notif_accum_year

## Transmission Variables

- receptivo
- transmissao

## Population Variables

- pop
- population_total
- population_density

## Demographic Variables

- male_population_pct
- female_population_pct
- aging_index
- median_age
- sex_ratio

## Meteorological Variables

- tempmin
- tempmed
- tempmax
- umidmin
- umidmed
- umidmax

## Precipitation Variables

- precip_total_semana
- precip_media_semana
- precip_max_semana
- dias_lluvia_semana

## Vegetation Variables

- NDVI_mean
- NDVI_std
- EVI_mean
- EVI_std

## Land Surface Temperature Variables

- LST_Day_mean
- LST_Day_std
- LST_Night_mean
- LST_Night_std

## Global Climate Variables

- nino34
- soi

## Topography Variables

- elev_mean
- elev_min
- elev_max

## Land-Cover Variables

- dominant_landcover
- pct_tree_cover
- pct_shrubland
- pct_grassland
- pct_cropland
- pct_builtup
- pct_water
- pct_wetland
- pct_mangroves
- pct_bare_sparse

## Auxiliary Spatial Variables

- n_pixels
- n_tiles

## Geographic Variables

- area_km2

## Sanitation Variables

- water_supply_pct
- sewage_pct
- garbage_collection_pct

## Temporal lag Variables

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

## Rolling Window Variables

- casos_roll4_mean
- casos_roll8_mean

- precip_roll4_sum
- precip_roll8_sum

- tempmed_roll4_mean
- tempmed_roll8_mean

---

# Variable Definitions

## casos

Reported dengue cases notified during the epidemiological week.

Unit: cases

Source: InfoDengue

## casos_est

Estimated dengue cases after statistical correction performed by InfoDengue.

Unit: cases

Source: InfoDengue

## p_inc100k

Estimated dengue incidence per 100,000 inhabitants.

Formula: (casos_est / population) × 100,000

Unit: cases per 100,000 inhabitants

Source: InfoDengue

## Rt

Effective reproduction number.

Represents the average number of secondary infections generated by an infected individual at a given time.

Interpretation:

- Rt > 1 : epidemic growth
- Rt = 1 : stable transmission
- Rt < 1 : epidemic decline

Unit: dimensionless

Source: InfoDengue

## population_density

Municipal population density.

Formula: population_total / area_km2

Unit: inhabitants per km²

Source: IBGE Census 2022 + IBGE municipal boundaries

## population_total

Total resident population of the municipality.

Reference year: 2022 Census

Unit: inhabitants

Source: IBGE Census 2022

## median_age

Median age of the resident population.

Represents the age that divides the population into two groups of equal size.

Unit: years

Source: IBGE Census 2022

## aging_index

Population aging index.

Represents the ratio between the elderly population and the young population.

Higher values indicate an older population structure.

Unit: dimensionless

Source: IBGE Census 2022

## water_supply_pct

Percentage of the municipal population with access to water supply services.

Unit: %

Source: Instituto Água e Saneamento

Interpretation: Higher values indicate better access to treated water infrastructure.

## sewage_pct

Percentage of the municipal population with access to sewage collection services.

Unit: %

Source: Instituto Água e Saneamento

Interpretation: Higher values indicate better sanitation coverage.

## garbage_collection_pct

Percentage of the municipal population covered by solid waste collection services.

Unit: %

Source: Instituto Água e Saneamento

Interpretation: Higher values indicate better waste management infrastructure.

## NDVI_mean

Mean Normalized Difference Vegetation Index for the municipality and month.

NDVI is a remote sensing indicator of vegetation presence and vigor.

Typical range: -1 to +1

Interpretation: Higher values indicate denser and healthier vegetation.

Source: MODIS Vegetation Indices

## EVI_mean

Mean Enhanced Vegetation Index for the municipality and month.

EVI improves vegetation monitoring in high-biomass regions and is less sensitive to atmospheric effects than NDVI.

Typical range: -1 to +1

Source: MODIS Vegetation Indices

## LST_Day_mean

Mean daytime Land Surface Temperature.

Represents the average temperature of the Earth's surface observed during the daytime satellite overpass.

Unit: °C

Source: MODIS Land Surface Temperature

## LST_Night_mean

Mean nighttime Land Surface Temperature.

Represents the average temperature of the Earth's surface observed during the nighttime satellite overpass.

Unit: °C

Source: MODIS Land Surface Temperature

## precip_total_semana

Total accumulated precipitation during the epidemiological week.

Unit: millimeters (mm)

Source: CHIRPS v2.0

Interpretation: Higher values indicate wetter environmental conditions and potential increases in mosquito breeding sites.

## tempmed

Mean air temperature during the epidemiological week.

Unit: °C

Source: InfoDengue meteorological data

Interpretation: Temperature strongly influences mosquito development, survival and virus replication.

## umidmed

Mean relative humidity during the epidemiological week.

Unit: %

Source: InfoDengue meteorological data

Interpretation: Higher humidity generally favors mosquito survival and activity.

## nino34

Sea Surface Temperature anomaly in the Niño 3.4 region of the equatorial Pacific Ocean.

This is one of the most widely used indicators of ENSO (El Niño-Southern Oscillation).

Interpretation:

- Positive values: El Niño conditions
- Negative values: La Niña conditions

Unit: °C anomaly

Source: NOAA Climate Prediction Center

## soi

Southern Oscillation Index.

Atmospheric component of ENSO based on pressure differences between Tahiti and Darwin.

Interpretation:

- Positive values: La Niña tendency
- Negative values: El Niño tendency

Unit: dimensionless

Source: NOAA Climate Prediction Center

---

# Missing Values Summary

Main variables with missing observations:

| Variable | Missing Records |
|-----------|----------------|
| sewage_pct | 2,403,752 |
| garbage_collection_pct | 607,555 |
| LST_Night_mean | 432,859 |
| LST_Night_std | 432,859 |
| water_supply_pct | 411,747 |
| umidmed | 403,920 |
| tempmax | 402,333 |
| tempmed | 402,333 |
| umidmin | 388,592 |
| umidmax | 334,782 |
| LST_Day_mean | 214,549 |
| LST_Day_std | 214,549 |

Notes:

- Missing values originate from source availability and quality filters.
- Missing values are not caused by the integration process.
- Municipal matching coverage is complete.

---

# Sanitation Coverage

| Variable | Approximate Coverage |
|-----------|---------------------|
| water_supply_pct | 91.2% |
| sewage_pct | 48.9% |
| garbage_collection_pct | 87.1% |

---

# Quality Assessment

Municipal coverage: 5570 / 5570 municipalities

Municipal matching rate: 100%

Duplicate municipality IDs: 0

Geographic consistency: Validated

Temporal consistency: Validated

Administrative consistency: Validated

---

# Known Limitations

1. Sewage coverage is limited by source availability.

2. Several environmental variables contain missing observations due to satellite quality filters.

3. Climatic variables obtained from InfoDengue contain missing observations for some municipalities and weeks.

4. Socioeconomic indicators such as income per capita and IDHM are not included in v8.

5. Temporal lag features are limited to a small set of epidemiological and climatic variables.

---

## Processing Scripts

### Geographic Base

- 00_build_municipality_master.py

### Epidemiology

- 01_download_infodengue_nacional.py
- 02_build_dataset_epidemiologico.py

### Precipitation

- 03_build_dataset_precipitacion_chirps_anual.py
- 04_merge_dataset_precipitacion.py
- 05_build_dataset_master_v1.py

### MODIS

- 06_build_datasets_modis.py
- 07_build_dataset_modis_master.py
- 08_build_dataset_modis_quality_report.py
- 09_merge_dataset_master_modis.py

### ENSO

- 10_build_dataset_enso.py
- 11_merge_enso.py

### Elevation

- 12_build_dem_tile_mapping.py
- 13_build_dataset_elevation.py
- 14_merge_elevation.py

### Land Cover

- 15_build_dataset_landcover.py
- 16_merge_landcover.py

### Demography

- 18_build_dataset_demographics.py
- 19_merge_demographics.py

### Population Density

- 20_build_population_density.py
- 21_merge_population_density.py

### Sanitation

- 22_build_sanitation.py
- 23_merge_sanitation.py

### Lag variables and temporal rolling windows

- 24_build_temporal_lags.py

---

# Intended Uses

This dataset is intended for:

- Dengue forecasting
- Dengue outbreak prediction
- Machine learning benchmarking
- Explainable AI (XAI)
- Climate-health studies
- Environmental epidemiology
- Spatial epidemiology
- Deep learning applications
- Epidemiology-informed AI systems

---

# Version History

## v1

Epidemiology + precipitation + MODIS

## v2

v1 + ENSO

## v3

v2 + elevation

## v4

v3 + land cover

## v5

v4 + demography

## v6

v5 + area_km2 + population_density

## v7

v6 + sanitation variables

## v8

v7 + temporal feature engineering

New lag variables:

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

New rolling window variables:

- casos_roll4_mean
- casos_roll8_mean

- precip_roll4_sum
- precip_roll8_sum

- tempmed_roll4_mean
- tempmed_roll8_mean

---

# Dataset Coverage Summary

Records: 4,701,298

Variables: 87

Municipalities: 5,570

States: 27

Temporal coverage: 2010–2025

Municipality matching: 100%

Duplicate municipality IDs: 0

---

Current status:

- Version: v8
- Stable baseline release
- Ready for exploratory analysis and benchmark modeling

Recommended use: This version should be considered the reference dataset for baseline analyses, benchmark evaluation and machine learning experiments.

Generation date: 2026-08-18

---

## Potential v9 Additions

- Additional temporal windows
- Socioeconomic indicators (IDHM)
- Income per capita
- Healthcare coverage
- Advanced engineered features

---

## Citation

If you use DATASET_MULTIMODAL_V8 in scientific work, please cite the corresponding dataset publication when available.

Citation details will be provided in a future release of this document.

---

## Contact

Project:

Early Detection of Dengue Outbreaks in Brazil

Corresponding author: J. Salvador Sánchez (sanchez@uji.es)

---

# License

Academic research use.
