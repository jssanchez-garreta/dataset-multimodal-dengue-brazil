
# BENCHMARK COMPARISON

## Forecasting Ranking

       Model  MAE_Mean    MAE_SD  RMSE_Mean   RMSE_SD  R2_Mean    R2_SD  Rank
    LightGBM 16.489253 10.064913  57.311596 30.727665 0.800963 0.047870     1
    CatBoost 16.629376 10.104170  57.528661 30.725602 0.799870 0.046748     2
     XGBoost 17.066025 10.869317  61.296381 34.669864 0.778439 0.044638     3
RandomForest 20.399808 14.514498  79.213970 51.691249 0.653533 0.079814     4

## Best Forecasting Model

LightGBM

R2 = 0.8010

## Nowcasting vs Forecasting

LightGBM Nowcasting:

R2 = 0.9869

LightGBM Forecasting:

R2 = 0.8010

Difference:

-0.1860

## Friedman Test

Statistic:

13.5600

p-value:

0.003570

## Interpretation

If p < 0.05:

Reject H0

At least one model behaves
significantly differently.

See pairwise_wilcoxon.csv
for pairwise comparisons.
