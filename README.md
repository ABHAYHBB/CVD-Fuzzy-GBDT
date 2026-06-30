# CVD-Fuzzy-GBDT

This repository contains Python code for cardiovascular disease prediction using fuzzy feature transformation, Gradient Boosting Decision Tree (GBDT), and Bagging GBDT models.

The code supports two datasets: the UCI Heart Disease dataset and the Framingham dataset. It includes scripts for fuzzy feature transformation, triangular fuzzy transformation, trapezoidal fuzzy transformation, GBDT training, Bagging GBDT training, model evaluation, standard cross-validation, nested cross-validation, ROC-AUC plotting, and result generation.

## Requirements

Install the required Python packages using:

pip install -r requirements.txt

Main packages used in this project are numpy, pandas, scipy, scikit-learn, matplotlib, seaborn, bokeh, openpyxl, and joblib.

## Dataset

For Dataset One, place the UCI Heart Disease dataset in the dataset_one_uci folder. The target column should be named target.

For Dataset Two, place the Framingham dataset in the dataset_two_framingham folder. The target column should be named TenYearCHD.

If the dataset column names are different, update the dataset path, target column, and selected feature names inside the Python scripts before running the code.

## Fuzzy transformation

The fuzzy transformation scripts convert selected numerical clinical features into transformed fuzzy features. The repository includes hybrid fuzzy transformation, triangular fuzzy transformation, and trapezoidal fuzzy transformation scripts.

The transformed dataset should be saved before running the model training scripts.

## Running the models

To run GBDT on the transformed dataset:

python code/train_gbdt_transformed.py

To run Bagging GBDT on the transformed dataset:

python code/train_bagging_gbdt_transformed.py

Before running each script, check the dataset path, target column name, transformed feature column names, and result folder.

## Evaluation

The models are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, Matthews correlation coefficient, and Cohen’s Kappa. The scripts also generate ROC-AUC plots, confusion matrices, cross-validation results, and CSV result tables.

## Nested cross-validation

The nested_cross_validation.py file provides a reusable function for formal nested cross-validation. The outer loop is used for model evaluation, and the inner loop is used for hyperparameter tuning.

## Output

The generated outputs include performance tables, ROC-AUC figures, standard cross-validation results, nested cross-validation summaries, and best hyperparameter records.

## Repository link

https://github.com/ABHAYHBB/CVD-Fuzzy-GBDT/tree/main
