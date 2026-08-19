# AUDIT_V8
Generated automatically from dataset_multimodal_v8.parquet.
## General Summary
- Records: 4,701,298
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
| sewage_pct | 2,403,751 | 51.13% |
| garbage_collection_pct | 607,555 | 12.92% |
| LST_Night_mean | 432,859 | 9.21% |
| LST_Night_std | 432,859 | 9.21% |
| umidmed_lag_4 | 426,187 | 9.07% |
| tempmed_lag_4 | 424,600 | 9.03% |
| water_supply_pct | 411,746 | 8.76% |
| umidmed_lag_1 | 409,486 | 8.71% |
| tempmed_lag_1 | 407,899 | 8.68% |
| umidmed | 403,919 | 8.59% |
| tempmax | 402,332 | 8.56% |
| tempmed | 402,332 | 8.56% |
| umidmin | 388,591 | 8.27% |
| tempmed_roll4_mean | 335,540 | 7.14% |
| umidmax | 334,781 | 7.12% |
| tempmed_roll8_mean | 292,955 | 6.23% |
| LST_Day_std | 214,549 | 4.56% |
| LST_Day_mean | 214,549 | 4.56% |
| tempmin | 205,551 | 4.37% |
| EVI_mean | 122,504 | 2.61% |

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
