import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    make_scorer
)



# 1. Basic settings

DATASET_PATH = "data/dataset_one_uci/heart_dataset_fuzzy.csv"

TARGET_COLUMN = "target"

# Use your actual transformed feature column here.
# Example: thalach_fuzzy, age_fuzzy, chol_fuzzy, sysBP_fuzzy, etc.
TRANSFORMED_FEATURE_COLUMNS = ["thalach_fuzzy"]

RESULT_FOLDER = "results/dataset_one_uci"

TEST_SIZE = 0.30
RANDOM_STATE = 42
CV_FOLDS = 10



# 2. Load dataset


df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

if TARGET_COLUMN not in df.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

for feature in TRANSFORMED_FEATURE_COLUMNS:
    if feature not in df.columns:
        raise ValueError(f"Transformed feature '{feature}' was not found.")



# 3. Select transformed feature and target
X = df[TRANSFORMED_FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

# If transformed feature is categorical, for example Low/Medium/High,
# it will be converted into numeric columns automatically.
X = pd.get_dummies(X, drop_first=False)

print("\nFeatures used for training:")
print(X.columns.tolist())

print("\nTarget distribution:")
print(y.value_counts())



# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)



# 5. Define GBDT model


gbdt_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=RANDOM_STATE
)



# 6. Train model


gbdt_model.fit(X_train, y_train)


# 7. Prediction on test data


y_pred = gbdt_model.predict(X_test)

# Probability of positive class, needed for ROC-AUC
y_prob = gbdt_model.predict_proba(X_test)[:, 1]



# 8. Performance metrics on test data


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_prob)
mcc = matthews_corrcoef(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("\n====================================")
print("GBDT Test Results")
print("====================================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print(f"MCC       : {mcc:.4f}")
print(f"Kappa     : {kappa:.4f}")

print("\nConfusion Matrix:")
print(cm)



# 9. Standard cross-validation


cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

scoring = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "mcc": make_scorer(matthews_corrcoef),
    "kappa": make_scorer(cohen_kappa_score)
}

cv_results = cross_validate(
    estimator=gbdt_model,
    X=X,
    y=y,
    cv=cv,
    scoring=scoring
)

print("\n====================================")
print("Standard 10-Fold Cross-Validation")
print("====================================")

cv_summary = {}

for metric in scoring.keys():
    values = cv_results[f"test_{metric}"]
    mean_value = np.mean(values)
    std_value = np.std(values)

    cv_summary[metric] = [mean_value, std_value]

    print(f"{metric.upper():10s}: {mean_value:.4f} ± {std_value:.4f}")



# 10. ROC-AUC plot


fpr, tpr, thresholds = roc_curve(y_test, y_prob)

figure_folder = os.path.join(RESULT_FOLDER, "figures")
os.makedirs(figure_folder, exist_ok=True)

roc_path = os.path.join(
    figure_folder,
    "gbdt_transformed_feature_roc_auc.png"
)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"GBDT, AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curve for GBDT Using Transformed Feature")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()

plt.savefig(roc_path, dpi=300)
plt.show()

print("\nROC-AUC plot saved at:")
print(roc_path)



# 11. Save result table


table_folder = os.path.join(RESULT_FOLDER, "performance_tables")
os.makedirs(table_folder, exist_ok=True)

result_table = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC",
        "MCC",
        "Kappa"
    ],
    "Test_Result": [
        accuracy,
        precision,
        recall,
        f1,
        auc,
        mcc,
        kappa
    ],
    "CV_Mean": [
        cv_summary["accuracy"][0],
        cv_summary["precision"][0],
        cv_summary["recall"][0],
        cv_summary["f1"][0],
        cv_summary["roc_auc"][0],
        cv_summary["mcc"][0],
        cv_summary["kappa"][0]
    ],
    "CV_Std": [
        cv_summary["accuracy"][1],
        cv_summary["precision"][1],
        cv_summary["recall"][1],
        cv_summary["f1"][1],
        cv_summary["roc_auc"][1],
        cv_summary["mcc"][1],
        cv_summary["kappa"][1]
    ]
})

result_path = os.path.join(
    table_folder,
    "gbdt_transformed_feature_results.csv"
)

result_table.to_csv(result_path, index=False)

print("\nResult table saved at:")
print(result_path)

print("\nFinal result table:")
print(result_table)
