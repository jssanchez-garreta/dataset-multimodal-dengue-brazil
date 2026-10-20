# AUDIT_V8
Generated automatically from dataset_multimodal_v8.parquet.
## General Summary
- Records: 4,634,540
- Variables: 87
- Municipalities: 5,570
- States: 27
- Temporal coverage: 2010-2025
- Duplicate rows: 0

## Data Types
- float64: 62
- int64: 12
- float32: 5
- object: 5
- int32: 2
- datetime64[ns]: 1

## Top Missing Values
| Variable | Missing | Missing % |
|----------|---------|-----------|
| sewage_pct | 2,369,620 | 51.13% |
| garbage_collection_pct | 598,927 | 12.92% |
| LST_Night_mean | 427,827 | 9.23% |
| LST_Night_std | 427,827 | 9.23% |
| umidmed_lag_4 | 419,301 | 9.05% |
| tempmed_lag_4 | 417,714  | 9.01% |
| water_supply_pct | 405,901 | 8.76% |

## Lag Variables
- Rt_lag_1
- Rt_lag_2
- Rt_lag_4
- casos_lag_1
- casos_lag_2
- casos_lag_4
- casos_lag_8
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

## Key Epidemiological Variables

### casos
- Mean: 7.8797
- Median: 0.0000
- Min: 0.0000
- Max: 85389.0000

### casos_est
- Mean: 7.8797
- Median: 0.0000
- Min: 0.0000
- Max: 85389.0000

### Rt
- Mean: 0.9910
- Median: 0.0000
- Min: 0.0000
- Max: 18.5216

### p_inc100k
- Mean: 19.2606
- Median: 0.0000
- Min: 0.0000
- Max: 7103.0640

## Audit Conclusion
The dataset passed the structural audit.
- 5,570 municipalities represented.
- 87 variables available.
- 13 lag variables generated.
- 6 rolling-window variables generated.
- Duplicate rows detected: 0.
