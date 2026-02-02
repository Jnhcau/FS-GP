# FS-GP
![Main Figure](https://github.com/Jnhcau/FS-GP/blob/main/image.jpg)

This repository provides the implementation for the study:

**"Benchmarking Feature Selection Methods and Prediction Models for Flowering Time Prediction in Plant Breeding" (IJMS)**

The code supports full reproducibility of all experiments reported in the manuscript.

## Contents

This repository includes:

1. Fold generation and nested cross-validation splits  
2. Feature selection (FS) methods  
3. Model training and hyperparameter tuning  
4. Model evaluation and performance analysis  
5. SHAP-based feature importance visualization

- feature_selectors.py  
  Implements all feature selection methods, including:
  ElasticNet, LASSO, Random Forest, XGBoost, LightGBM, Mutual Information, and Boruta.

- Prediction.py  
  Main script for running the nested cross-validation pipeline.
  It calls feature selection methods from `feature_selectors.py`,
  performs model training, hyperparameter tuning, and evaluation.

- SHAP.py  
  Script for computing and visualizing feature importance using SHAP values.
