import os
import numpy as np
import pandas as pd

from scipy import stats

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    cohen_kappa_score
)



# Nested cross-validation function


def run_nested_cross_validation(
    model,
    param_grid,
    X,
    y,
    outer_folds=10,
    inner_folds=5,
    scoring_for_tuning="roc_auc",
    random_state=42,
    save_results=True,
    output_folder="results/nested_cv",
    output_file_name="nested_cv_results.csv"
):
    """
    Formal nested cross-validation.

    Outer loop:
        Used only for final model evaluation.

    Inner loop:
        Used only for hyperparameter selection through GridSearchCV.

    Parameters
    ----------
    model:
        Any sklearn-compatible classifier.
        Example: GBDT model or Bagging GBDT model already defined outside.

    param_grid:
        Hyperparameter grid for GridSearchCV.

    X:
        Feature matrix.
        This should already contain transformed features.

    y:
        Target labels.

    outer_folds:
        Number of outer CV folds for final evaluation.

    inner_folds:
        Number of inner CV folds for hyperparameter tuning.

    scoring_for_tuning:
        Metric used inside GridSearchCV.
        Common options: "roc_auc", "accuracy", "f1".

    random_state:
        Random seed for reproducibility.

    save_results:
        If True, saves fold-wise and summary results as CSV files.

    output_folder:
        Folder where nested CV results will be saved.

    output_file_name:
        Name of the fold-wise nested CV result file.

    Returns
    -------
    fold_results_df:
        Fold-wise nested CV results.

    summary_df:
        Mean, standard deviation, and 95% confidence interval.

    best_params_df:
        Best hyperparameters selected in each outer fold.
    """

    outer_cv = StratifiedKFold(
        n_splits=outer_folds,
        shuffle=True,
        random_state=random_state
    )

    inner_cv = StratifiedKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=random_state
    )

    fold_results = []
    best_params_list = []

    print("\n====================================")
    print("Nested Cross-Validation Started")
    print("====================================")
    print("Outer folds:", outer_folds)
    print("Inner folds:", inner_folds)
    print("Tuning metric:", scoring_for_tuning)

    for fold_number, (train_index, test_index) in enumerate(
        outer_cv.split(X, y),
        start=1
    ):

        print("\n------------------------------------")
        print("Outer Fold:", fold_number)
        print("------------------------------------")

        X_outer_train = X.iloc[train_index]
        X_outer_test = X.iloc[test_index]

        y_outer_train = y.iloc[train_index]
        y_outer_test = y.iloc[test_index]

        # Fresh copy of the model for every outer fold
        current_model = clone(model)

        # Inner loop: hyperparameter tuning only on outer-training data
        grid_search = GridSearchCV(
            estimator=current_model,
            param_grid=param_grid,
            scoring=scoring_for_tuning,
            cv=inner_cv,
            n_jobs=-1,
            refit=True
        )

        grid_search.fit(X_outer_train, y_outer_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

        best_params_list.append({
            "Fold": fold_number,
            "Best_Params": best_params
        })

        # Outer loop: final evaluation only on held-out outer-test fold
        y_pred = best_model.predict(X_outer_test)

        # ROC-AUC needs probability scores.
        # This keeps the code usable for models with predict_proba or decision_function.
        if hasattr(best_model, "predict_proba"):
            y_score = best_model.predict_proba(X_outer_test)[:, 1]
        elif hasattr(best_model, "decision_function"):
            y_score = best_model.decision_function(X_outer_test)
        else:
            y_score = y_pred

        accuracy = accuracy_score(y_outer_test, y_pred)
        precision = precision_score(y_outer_test, y_pred, zero_division=0)
        recall = recall_score(y_outer_test, y_pred, zero_division=0)
        f1 = f1_score(y_outer_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_outer_test, y_score)
        mcc = matthews_corrcoef(y_outer_test, y_pred)
        kappa = cohen_kappa_score(y_outer_test, y_pred)

        fold_results.append({
            "Fold": fold_number,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_score": f1,
            "ROC_AUC": roc_auc,
            "MCC": mcc,
            "Kappa": kappa
        })

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1-score  : {f1:.4f}")
        print(f"ROC-AUC   : {roc_auc:.4f}")
        print(f"MCC       : {mcc:.4f}")
        print(f"Kappa     : {kappa:.4f}")
        print("Best parameters:", best_params)

    fold_results_df = pd.DataFrame(fold_results)
    best_params_df = pd.DataFrame(best_params_list)

    summary_df = create_nested_cv_summary(fold_results_df)

    print("\n====================================")
    print("Nested Cross-Validation Summary")
    print("====================================")
    print(summary_df)

    if save_results:
        os.makedirs(output_folder, exist_ok=True)

        fold_path = os.path.join(output_folder, output_file_name)
        summary_path = os.path.join(output_folder, "nested_cv_summary.csv")
        best_params_path = os.path.join(output_folder, "nested_cv_best_params.csv")

        fold_results_df.to_csv(fold_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        best_params_df.to_csv(best_params_path, index=False)

        print("\nSaved files:")
        print(fold_path)
        print(summary_path)
        print(best_params_path)

    return fold_results_df, summary_df, best_params_df



# Summary table with 95% confidence interval


def create_nested_cv_summary(fold_results_df):
    """
    Creates mean, standard deviation, and 95% confidence interval
    from outer-fold nested CV results.
    """

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_score",
        "ROC_AUC",
        "MCC",
        "Kappa"
    ]

    summary_rows = []

    n_folds = len(fold_results_df)

    for metric in metrics:
        values = fold_results_df[metric].values

        mean_value = np.mean(values)
        std_value = np.std(values, ddof=1)

        # 95% confidence interval using t-distribution
        standard_error = std_value / np.sqrt(n_folds)
        t_value = stats.t.ppf(0.975, df=n_folds - 1)
        ci_margin = t_value * standard_error

        ci_lower = mean_value - ci_margin
        ci_upper = mean_value + ci_margin

        summary_rows.append({
            "Metric": metric,
            "Mean": mean_value,
            "Std": std_value,
            "CI_95_Lower": ci_lower,
            "CI_95_Upper": ci_upper,
            "Mean_Percentage": mean_value * 100,
            "Std_Percentage": std_value * 100,
            "CI_95_Lower_Percentage": ci_lower * 100,
            "CI_95_Upper_Percentage": ci_upper * 100
        })

    summary_df = pd.DataFrame(summary_rows)

    return summary_df
