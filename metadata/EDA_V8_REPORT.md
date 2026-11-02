
# EDA_V8_REPORT

## Dataset Summary

- Records: 4,701,298
- Variables: 87
- Municipalities: 5,570
- States: 27
- Temporal Coverage: 2010-2025
- Duplicate Rows: 0

## Generated Figures

- missing_values.png
- dengue_timeseries_brazil.png
- climate_distributions.png
- sanitation_distributions.png
- demographic_distributions.png
- lag_distributions.png
- correlation_matrix.png

## Generated CSV Tables

- dataset_overview.csv
- missing_values.csv
- correlation_ranking.csv  -> top 30 correlations
- all_correlations.csv     -> complete ranking
- feature_groups.csv

## Preliminary Assessment

The dataset appears structurally consistent
and suitable for benchmark machine learning
experiments.

Important missing values are concentrated in:

- sewage_pct
- sanitation variables
- LST variables
- humidity and temperature variables

Temporal feature engineering was successfully
integrated through lag and rolling features.
