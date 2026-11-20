# Benchmark Experiments

This directory contains scripts used to construct benchmark datasets and evaluate machine learning models for dengue nowcasting and forecasting.

Evaluated models:

- Naive Persistence
- Ridge Regression
- LinearSVR
- LightGBM
- XGBoost
- CatBoost
- Random Forest
- 5-Layer MLP
- Cascade Forest
- HAE (Horizon Adaptive Multimodal Ensemble)

Included tasks:

- Benchmark dataset creation
- Model training
- Temporal cross-validation
- Statistical model comparison

Main scripts:

- 28_build_benchmark_dataset.py
- 29_benchmark_lightgbm.py
- 30_benchmark_xgboost.py
- 31_benchmark_catboost.py
- 32_benchmark_random_forest.py
- 33_benchmark_deep_forest_simplified.py
- 34_benchmark_ridge.py
- 35_becnhmark_naive_persistence.py
- 36_bnchmark_svr.py
- 37_benchmark_mlp_layers.py
- 38_benchmark_mlp_v2.py
- 40_compare_models.py
- 51_statistical_comparison.py
- 52_extended_scientific_experiments.py
- 53_horizon_adaptive_ensemble.py
