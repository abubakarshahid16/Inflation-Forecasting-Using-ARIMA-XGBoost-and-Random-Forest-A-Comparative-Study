# Inflation Forecasting Comparative Study

A comparative forecasting project that studies how classical time-series modeling and machine-learning methods perform on inflation prediction.

This repository is best understood as an **economic forecasting comparison workflow** with supporting visual analysis, model outputs, and written documentation.

## Problem this project solves

Inflation forecasting matters because it affects:

- monetary-policy planning
- financial analysis
- budgeting assumptions
- pricing and demand expectations
- macroeconomic decision-making

No single model family is always best, so this project compares statistical and machine-learning approaches to understand how their behavior differs on the same problem.

## What this project does

The repository compares:

- ARIMA for classical time-series forecasting
- XGBoost for boosted tree-based prediction
- Random Forest for nonlinear ensemble forecasting

The goal is not just to produce one forecast, but to evaluate which modeling style behaves better under the available dataset and evaluation setup.

## Visual highlights

| Model Comparison | ARIMA Performance |
| --- | --- |
| ![Model comparison](mse_comparison.png) | ![ARIMA performance](arima_performance.png) |

| Random Forest Importance | XGBoost Importance |
| --- | --- |
| ![Random Forest feature importance](rf_variable_importance.png) | ![XGBoost feature importance](xgb_feature_importance.png) |

## Repository contents

- `project_code.r`: main implementation
- `project_documentation.pdf`: project report
- `dataset (2).csv`: dataset used in the analysis
- `model_comparison.csv`: summarized model results
- supporting plots for correlation, error comparison, feature importance, and distributions

## Analysis areas

- model error comparison
- inflation trend forecasting
- correlation analysis
- distribution and boxplot exploration
- training and testing performance review
- variable-importance interpretation for tree-based methods

## Why this project matters

- It demonstrates economic forecasting with both classical and ML methods.
- It compares models instead of assuming one method is best.
- It provides evidence through plots and tabular outputs.
- It is useful as a portfolio project for forecasting, analytics, and applied econometrics.

## Run locally

Open the main script:

```text
project_code.r
```

Typical workflow:

1. open the R script in RStudio or another R environment
2. load the dataset
3. run preprocessing and model sections
4. regenerate plots and comparison outputs

## Industrial positioning

In a more production-style setting, an inflation forecasting workflow would also need:

- regular data refresh pipelines
- rolling backtests
- forecast monitoring over time
- feature-governance and data-quality checks
- scenario analysis for policy or planning teams

This makes the repo best positioned as a **comparative macroeconomic forecasting study** with practical analytical value.
