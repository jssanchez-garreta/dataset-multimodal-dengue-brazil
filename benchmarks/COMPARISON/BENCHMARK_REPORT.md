
# BENCHMARK COMPARISON

## Forecasting Ranking

       Model  MAE_Mean    MAE_SD  RMSE_Mean   RMSE_SD  R2_Mean    R2_SD  Rank
    LightGBM 17.605912 10.650096  59.905905 31.663866 0.784342 0.044768     1
    CatBoost 17.750890 10.712512  60.242034 32.071966 0.782020 0.048007     2
     XGBoost 18.122181 11.659705  63.433657 36.297204 0.767532 0.036085     3
RandomForest 21.029960 14.851787  79.803437 50.729208 0.644676 0.076555     4

## Best Forecasting Model

LightGBM

R2 = 0.7843

## Nowcasting vs Forecasting

LightGBM Nowcasting:

R2 = 0.9869

LightGBM Forecasting:

R2 = 0.7843

Difference:

-0.2026

## Friedman Test

Statistic:

12.8400

p-value:

0.004996

## Interpretation

If p < 0.05:

Reject H0

At least one model behaves
significantly differently.

See pairwise_wilcoxon.csv
for pairwise comparisons.
