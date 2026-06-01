# 06_compare_all_models_level2.py

"""
Compare three machine learning model families for PBMC68k cell type prediction
using the intermediate hierarchical label level: cell_type_level_2.

This script compares 6 configurations:
1. Logistic Regression without class weighting
2. Logistic Regression with balanced class weighting
3. Linear SVM without class weighting
4. Linear SVM with balanced class weighting
5. Random Forest without class weighting
6. Random Forest with balanced class weighting

Evaluation design:
- Reuses the same 80/20 development-test split created during the Level 1 experiments
  when the split file is available.
- Performs 5-fold stratified cross-validation on the development set.
- Also trains each model on the full development set and evaluates it once on the
  untouched final test set. Use CV results for model comparison/model selection;
  use test results only as the final held-out confirmation.

Outputs:
- One Excel workbook with all result tables.
- One metadata JSON file.
- One split index file for reproducibility.
- Heatmaps for summed CV confusion matrices and final test confusion matrices.
- A compact presentation summary table image.
- Metric barplots for Macro F1, Weighted F1, and Balanced Accuracy.

Dataset:
pbmc68k_preprocessed_for_training.h5ad

Label:
cell_type_level_2
"""

from pathlib import Path
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss,
    roc_auc_score,
)


# Basic settings

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20
LABEL_COLUMN = "cell_type_level_2"

DATA_PATH = Path(
    r"C:\Users\ferid\Downloads\capstone_demo\outputs\pbmc68k_preprocessed_for_training.h5ad"
)

PROJECT_DIR = Path(r"C:\Users\ferid\Downloads\capstone_demo\model_development")

RESULTS_DIR = PROJECT_DIR / "results" / "all_models_level2"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_PATH = RESULTS_DIR / "all_models_level2_CV_results.xlsx"
METADATA_PATH = RESULTS_DIR / "all_models_level2_metadata.json"
SPLIT_PATH = RESULTS_DIR / "all_models_level2_split_indices.npz"

# Reuse the previous Level 1 LR split to keep Level 1 and Level 2 comparisons consistent.
PREVIOUS_SPLIT_PATH = (
    PROJECT_DIR
    / "results"
    / "LR_level1"
    / "LR_level1_split_indices.npz"
)


 
# Model constructors

def make_lr_model(class_weight):
    return LogisticRegression(
        max_iter=1000,
        solver="saga",
        penalty="l2",
        C=1.0,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_svm_model(class_weight):
    return LinearSVC(
        C=1.0,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        max_iter=5000,
        dual="auto",
    )


def make_rf_model(class_weight):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


 
# Helper functions
 
def calculate_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
    }

    if y_proba is not None:
        try:
            metrics["log_loss"] = log_loss(y_true, y_proba)
        except Exception:
            metrics["log_loss"] = np.nan

        try:
            metrics["roc_auc_macro_ovr"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        except Exception:
            metrics["roc_auc_macro_ovr"] = np.nan

        try:
            metrics["roc_auc_weighted_ovr"] = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="weighted",
            )
        except Exception:
            metrics["roc_auc_weighted_ovr"] = np.nan
    else:
        metrics["log_loss"] = np.nan
        metrics["roc_auc_macro_ovr"] = np.nan
        metrics["roc_auc_weighted_ovr"] = np.nan

    return metrics


def create_label_distribution_table(labels, table_name):
    counts = pd.Series(labels).value_counts().sort_index()
    percentages = counts / counts.sum() * 100
    df = pd.DataFrame({
        "class_name": counts.index,
        "count": counts.values,
        "percentage": percentages.values,
    })
    df.insert(0, "set_name", table_name)
    return df


def create_summary_table(fold_metrics_df, n_features, n_classes, n_dev, n_test):
    excluded_cols = {
        "model_name",
        "algorithm",
        "class_weight",
        "fold",
        "n_train_fold",
        "n_validation_fold",
    }
    metric_columns = [col for col in fold_metrics_df.columns if col not in excluded_cols]

    summary_rows = []
    for model_name in fold_metrics_df["model_name"].unique():
        model_df = fold_metrics_df[fold_metrics_df["model_name"] == model_name]
        row = {
            "model_name": model_name,
            "algorithm": model_df["algorithm"].iloc[0],
            "label_column": LABEL_COLUMN,
            "class_weight": model_df["class_weight"].iloc[0],
            "n_features": n_features,
            "n_classes": n_classes,
            "n_splits_cv": N_SPLITS,
            "development_set_size": n_dev,
            "test_set_size": n_test,
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = model_df[metric].mean()
            row[f"{metric}_std"] = model_df[metric].std()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if "f1_macro_mean" in summary_df.columns:
        summary_df = summary_df.sort_values("f1_macro_mean", ascending=False)
    return summary_df


def average_classification_reports(report_df, class_names):
    rows = []
    for model_name in report_df["model_name"].unique():
        model_part = report_df[report_df["model_name"] == model_name]
        for class_name in class_names:
            class_part = model_part[model_part["class_name"] == class_name]
            rows.append({
                "model_name": model_name,
                "class_name": class_name,
                "precision_mean": class_part["precision"].mean(),
                "precision_std": class_part["precision"].std(),
                "recall_mean": class_part["recall"].mean(),
                "recall_std": class_part["recall"].std(),
                "f1_score_mean": class_part["f1-score"].mean(),
                "f1_score_std": class_part["f1-score"].std(),
                "support_mean": class_part["support"].mean(),
                "support_sum": class_part["support"].sum(),
            })
    return pd.DataFrame(rows)


def normalize_confusion_matrix_rows(cm_df):
    cm_values = cm_df.values.astype(float)
    row_sums = cm_values.sum(axis=1, keepdims=True)
    normalized_values = np.divide(
        cm_values,
        row_sums,
        out=np.zeros_like(cm_values, dtype=float),
        where=row_sums != 0,
    )
    return pd.DataFrame(normalized_values, index=cm_df.index, columns=cm_df.columns)


def save_confusion_matrix_heatmap(cm_df, title, output_path, normalized=False):
    plot_df = normalize_confusion_matrix_rows(cm_df) if normalized else cm_df.copy()
    values = plot_df.values.astype(float)

    fig_width = max(8, len(plot_df.columns) * 1.25)
    fig_height = max(6, len(plot_df.index) * 1.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(values, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=13, pad=14)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(plot_df.columns)))
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_xticklabels(plot_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(plot_df.index)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_value = f"{values[i, j] * 100:.1f}%" if normalized else f"{int(values[i, j])}"
            ax.text(j, i, text_value, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_metric_barplot(summary_df, metric_col, title, output_path):
    plot_df = summary_df[["model_name", metric_col]].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot_df["model_name"], plot_df[metric_col])
    ax.set_title(title)
    ax.set_ylabel(metric_col)
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)

    for i, value in enumerate(plot_df[metric_col]):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_presentation_summary_table(summary_df, output_path):
    important_cols = [
        "model_name",
        "class_weight",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "f1_macro_mean",
        "f1_weighted_mean",
        "recall_macro_mean",
        "training_time_seconds_mean",
    ]
    existing_cols = [col for col in important_cols if col in summary_df.columns]
    table_df = summary_df[existing_cols].copy()

    table_df = table_df.rename(columns={
        "model_name": "Model",
        "class_weight": "Class Weight",
        "accuracy_mean": "Accuracy",
        "balanced_accuracy_mean": "Balanced Acc.",
        "f1_macro_mean": "Macro F1",
        "f1_weighted_mean": "Weighted F1",
        "recall_macro_mean": "Macro Recall",
        "training_time_seconds_mean": "Train Time (s)",
    })

    for col in table_df.columns:
        if col not in ["Model", "Class Weight"]:
            table_df[col] = table_df[col].apply(
                lambda x: f"{x:.3f}" if isinstance(x, (int, float, np.floating)) else x
            )

    fig, ax = plt.subplots(figsize=(13, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.45)
    ax.set_title("Level 2 Model Comparison: 5-Fold CV Summary", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def safe_excel_sheet_name(name):
    invalid_chars = ["\\", "/", "?", "*", "[", "]", ":"]
    for ch in invalid_chars:
        name = name.replace(ch, "_")
    return name[:31]


def save_metadata(metadata, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def get_predict_proba_if_available(model, X_values):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_values)
    return None


 
# Load dataset

print("Loading dataset...")
adata = sc.read_h5ad(DATA_PATH)
print("Dataset loaded:")
print(adata)

if LABEL_COLUMN not in adata.obs.columns:
    raise ValueError(f"Label column '{LABEL_COLUMN}' was not found in adata.obs.")

X = adata.X
if sparse.issparse(X):
    X = X.tocsr()

y_text = adata.obs[LABEL_COLUMN].astype(str).values

N_CELLS = adata.n_obs
N_FEATURES = adata.n_vars

print(f"\nNumber of cells: {N_CELLS}")
print(f"Number of features/HVGs: {N_FEATURES}")
print(f"Label column: {LABEL_COLUMN}")


 
# Encode labels

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)
CLASS_NAMES = list(label_encoder.classes_)
N_CLASSES = len(CLASS_NAMES)

print("\nEncoded classes:")
for encoded_label, class_name in enumerate(CLASS_NAMES):
    print(f"{encoded_label}: {class_name}")


 
# Reuse the previous Level 1 LR development-test split

loaded_previous_split = False
previous_split_used = None

if PREVIOUS_SPLIT_PATH.exists():
    previous_split = np.load(PREVIOUS_SPLIT_PATH)
    dev_indices = previous_split["dev_indices"]
    test_indices = previous_split["test_indices"]
    loaded_previous_split = True
    previous_split_used = PREVIOUS_SPLIT_PATH
    print(f"Loaded previous Level 1 split from: {PREVIOUS_SPLIT_PATH}")
else:
    raise FileNotFoundError(f"Previous Level 1 split file was not found:{PREVIOUS_SPLIT_PATH}"
        "Please run the Level 1 Logistic Regression comparison script first."
    )

# Check that development and test indices are valid.
if len(set(dev_indices).intersection(set(test_indices))) > 0:
    raise ValueError("Development and test indices overlap. Please check the split file.")

if max(dev_indices.max(), test_indices.max()) >= N_CELLS:
    raise ValueError("Split indices contain values outside the current dataset size.")

X_dev = X[dev_indices]
y_dev = y[dev_indices]
y_dev_text = label_encoder.inverse_transform(y_dev)

X_test = X[test_indices]
y_test = y[test_indices]
y_test_text = label_encoder.inverse_transform(y_test)

N_DEV = len(dev_indices)
N_TEST = len(test_indices)

np.savez(
    SPLIT_PATH,
    dev_indices=dev_indices,
    test_indices=test_indices,
)

print(f"Saved Level 2 split indices to: {SPLIT_PATH}")
print(f"Development set size: {N_DEV}")
print(f"Test set size: {N_TEST}")

# Warn if reused Level 1 split is not sufficiently stratified for Level 2 classes.
dev_counts_check = pd.Series(y_dev_text).value_counts()
if dev_counts_check.min() < N_SPLITS:
    raise ValueError(
        f"At least one Level 2 class has fewer than {N_SPLITS} cells in the development set. "
        "5-fold stratified CV is not valid for this label level."
    )


 
# Label distribution and class info tables
 
label_distribution_total = create_label_distribution_table(y_text, "total_dataset")
label_distribution_development_set = create_label_distribution_table(y_dev_text, "development_set_used_for_cv")
label_distribution_test_set = create_label_distribution_table(y_test_text, "final_test_set")

total_counts = pd.Series(y_text).value_counts().sort_index()
dev_counts = pd.Series(y_dev_text).value_counts().sort_index()
test_counts = pd.Series(y_test_text).value_counts().sort_index()

class_label_info = pd.DataFrame({
    "encoded_label": np.arange(N_CLASSES),
    "class_name": CLASS_NAMES,
    "total_count": [int(total_counts.get(c, 0)) for c in CLASS_NAMES],
    "development_count": [int(dev_counts.get(c, 0)) for c in CLASS_NAMES],
    "test_count": [int(test_counts.get(c, 0)) for c in CLASS_NAMES],
})


 
# Define candidate models
 
models = {
    "LR_no_weight": {
        "model": make_lr_model(class_weight=None),
        "algorithm": "Logistic Regression",
        "class_weight": "None",
    },
    "LR_balanced": {
        "model": make_lr_model(class_weight="balanced"),
        "algorithm": "Logistic Regression",
        "class_weight": "balanced",
    },
    "SVM_no_weight": {
        "model": make_svm_model(class_weight=None),
        "algorithm": "Linear SVM",
        "class_weight": "None",
    },
    "SVM_balanced": {
        "model": make_svm_model(class_weight="balanced"),
        "algorithm": "Linear SVM",
        "class_weight": "balanced",
    },
    "RF_no_weight": {
        "model": make_rf_model(class_weight=None),
        "algorithm": "Random Forest",
        "class_weight": "None",
    },
    "RF_balanced": {
        "model": make_rf_model(class_weight="balanced"),
        "algorithm": "Random Forest",
        "class_weight": "balanced",
    },
}

model_parameters_rows = []
for model_name, model_info in models.items():
    params = model_info["model"].get_params()
    row = {
        "model_name": model_name,
        "algorithm": model_info["algorithm"],
        "class_weight": str(params.get("class_weight")),
    }
    for key, value in params.items():
        if key not in row:
            row[key] = value
    model_parameters_rows.append(row)

model_parameters = pd.DataFrame(model_parameters_rows)


 
# Cross-validation on development set
 
cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_metrics_rows = []
per_fold_report_rows = []
cv_confusion_matrices = {}

print("\nStarting 5-fold stratified cross-validation on development set...")

for model_name, model_info in models.items():
    base_model = model_info["model"]
    algorithm = model_info["algorithm"]
    class_weight = model_info["class_weight"]

    print(f"\nModel: {model_name}")
    model_fold_cms = []

    for fold_number, (train_idx, val_idx) in enumerate(cv.split(X_dev, y_dev), start=1):
        print(f"  Fold {fold_number}/{N_SPLITS}")

        model = clone(base_model)

        X_train_fold = X_dev[train_idx]
        y_train_fold = y_dev[train_idx]
        X_val_fold = X_dev[val_idx]
        y_val_fold = y_dev[val_idx]

        fold_start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_fold, y_train_fold)
        training_time_seconds = time.time() - fold_start_time

        y_pred = model.predict(X_val_fold)
        y_proba = get_predict_proba_if_available(model, X_val_fold)

        metrics = calculate_metrics(y_true=y_val_fold, y_pred=y_pred, y_proba=y_proba)
        metrics.update({
            "model_name": model_name,
            "algorithm": algorithm,
            "class_weight": class_weight,
            "fold": fold_number,
            "training_time_seconds": training_time_seconds,
            "n_train_fold": len(train_idx),
            "n_validation_fold": len(val_idx),
        })
        fold_metrics_rows.append(metrics)

        cm = confusion_matrix(y_val_fold, y_pred, labels=np.arange(N_CLASSES))
        model_fold_cms.append(cm)

        report = classification_report(
            y_val_fold,
            y_pred,
            labels=np.arange(N_CLASSES),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        for class_name in CLASS_NAMES:
            per_fold_report_rows.append({
                "model_name": model_name,
                "algorithm": algorithm,
                "class_weight": class_weight,
                "fold": fold_number,
                "class_name": class_name,
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1-score": report[class_name]["f1-score"],
                "support": report[class_name]["support"],
            })

    summed_cm = np.sum(model_fold_cms, axis=0)
    cv_confusion_matrices[model_name] = pd.DataFrame(
        summed_cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )


 
# Final held-out test evaluation for each candidate model

print("\nTraining each model on full development set and evaluating on held-out test set...")

test_metrics_rows = []
test_report_rows = []
test_confusion_matrices = {}

for model_name, model_info in models.items():
    print(f"\nFinal test evaluation: {model_name}")
    model = clone(model_info["model"])
    algorithm = model_info["algorithm"]
    class_weight = model_info["class_weight"]

    train_start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_dev, y_dev)
    training_time_seconds = time.time() - train_start_time

    pred_start_time = time.time()
    y_test_pred = model.predict(X_test)
    prediction_time_seconds = time.time() - pred_start_time
    y_test_proba = get_predict_proba_if_available(model, X_test)

    metrics = calculate_metrics(y_true=y_test, y_pred=y_test_pred, y_proba=y_test_proba)
    metrics.update({
        "model_name": model_name,
        "algorithm": algorithm,
        "class_weight": class_weight,
        "training_time_seconds": training_time_seconds,
        "prediction_time_seconds": prediction_time_seconds,
        "n_train_development": N_DEV,
        "n_test": N_TEST,
    })
    test_metrics_rows.append(metrics)

    test_cm = confusion_matrix(y_test, y_test_pred, labels=np.arange(N_CLASSES))
    test_confusion_matrices[model_name] = pd.DataFrame(
        test_cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    report = classification_report(
        y_test,
        y_test_pred,
        labels=np.arange(N_CLASSES),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    for class_name in CLASS_NAMES:
        test_report_rows.append({
            "model_name": model_name,
            "algorithm": algorithm,
            "class_weight": class_weight,
            "class_name": class_name,
            "precision": report[class_name]["precision"],
            "recall": report[class_name]["recall"],
            "f1-score": report[class_name]["f1-score"],
            "support": report[class_name]["support"],
        })


 
# Create final result tables
 
fold_metrics = pd.DataFrame(fold_metrics_rows)
first_cols = [
    "model_name",
    "algorithm",
    "class_weight",
    "fold",
    "training_time_seconds",
    "n_train_fold",
    "n_validation_fold",
]
remaining_cols = [col for col in fold_metrics.columns if col not in first_cols]
fold_metrics = fold_metrics[first_cols + remaining_cols]

summary_metrics = create_summary_table(
    fold_metrics_df=fold_metrics,
    n_features=N_FEATURES,
    n_classes=N_CLASSES,
    n_dev=N_DEV,
    n_test=N_TEST,
)

per_fold_class_reports = pd.DataFrame(per_fold_report_rows)
average_class_reports = average_classification_reports(per_fold_class_reports, CLASS_NAMES)

test_metrics = pd.DataFrame(test_metrics_rows)
test_first_cols = [
    "model_name",
    "algorithm",
    "class_weight",
    "training_time_seconds",
    "prediction_time_seconds",
    "n_train_development",
    "n_test",
]
test_remaining_cols = [col for col in test_metrics.columns if col not in test_first_cols]
test_metrics = test_metrics[test_first_cols + test_remaining_cols]
if "f1_macro" in test_metrics.columns:
    test_metrics = test_metrics.sort_values("f1_macro", ascending=False)

test_class_reports = pd.DataFrame(test_report_rows)

presentation_summary = summary_metrics[[
    "model_name",
    "algorithm",
    "class_weight",
    "accuracy_mean",
    "balanced_accuracy_mean",
    "f1_macro_mean",
    "f1_weighted_mean",
    "recall_macro_mean",
    "training_time_seconds_mean",
]].copy()

presentation_summary = presentation_summary.rename(columns={
    "model_name": "Model",
    "algorithm": "Algorithm",
    "class_weight": "Class Weight",
    "accuracy_mean": "Accuracy (CV mean)",
    "balanced_accuracy_mean": "Balanced Accuracy (CV mean)",
    "f1_macro_mean": "Macro F1 (CV mean)",
    "f1_weighted_mean": "Weighted F1 (CV mean)",
    "recall_macro_mean": "Macro Recall (CV mean)",
    "training_time_seconds_mean": "Training Time s (CV mean)",
})

experiment_info = pd.DataFrame({
    "field": [
        "experiment_name",
        "date_time",
        "dataset_path",
        "label_column",
        "n_cells_total",
        "n_features",
        "n_classes",
        "class_names",
        "test_size",
        "development_set_size",
        "test_set_size",
        "previous_level1_split_reused",
        "previous_split_path",
        "cv_type",
        "n_splits_cv",
        "random_state",
        "models_compared",
        "model_selection_metric_recommended",
        "test_set_usage_note",
    ],
    "value": [
        "all_models_level2_CV",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        str(DATA_PATH),
        LABEL_COLUMN,
        N_CELLS,
        N_FEATURES,
        N_CLASSES,
        ", ".join(CLASS_NAMES),
        TEST_SIZE,
        N_DEV,
        N_TEST,
        loaded_previous_split,
        str(previous_split_used) if previous_split_used else "new split created",
        "StratifiedKFold on development set",
        N_SPLITS,
        RANDOM_STATE,
        ", ".join(models.keys()),
        "f1_macro_mean",
        "Use CV results for model selection; use final_test_metrics only as held-out confirmation.",
    ],
})


 
# Save metadata JSON
 
metadata = {
    "experiment_name": "all_models_level2_CV",
    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_path": str(DATA_PATH),
    "label_column": LABEL_COLUMN,
    "n_cells_total": int(N_CELLS),
    "n_features": int(N_FEATURES),
    "n_classes": int(N_CLASSES),
    "class_names": CLASS_NAMES,
    "test_size": TEST_SIZE,
    "development_set_size": int(N_DEV),
    "test_set_size": int(N_TEST),
    "previous_level1_split_reused": bool(loaded_previous_split),
    "previous_split_path": str(previous_split_used) if previous_split_used else None,
    "cv_type": "StratifiedKFold",
    "n_splits_cv": N_SPLITS,
    "random_state": RANDOM_STATE,
    "models_compared": list(models.keys()),
    "model_parameters": model_parameters.to_dict(orient="records"),
    "model_selection_metric_recommended": "f1_macro_mean",
}
save_metadata(metadata, METADATA_PATH)


 
# Save one Excel workbook with all tables
 
with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
    summary_metrics.to_excel(writer, sheet_name="summary_metrics", index=False)
    presentation_summary.to_excel(writer, sheet_name="presentation_summary", index=False)
    fold_metrics.to_excel(writer, sheet_name="fold_metrics", index=False)
    test_metrics.to_excel(writer, sheet_name="final_test_metrics", index=False)

    label_distribution_total.to_excel(writer, sheet_name="label_dist_total", index=False)
    label_distribution_development_set.to_excel(writer, sheet_name="label_dist_dev", index=False)
    label_distribution_test_set.to_excel(writer, sheet_name="label_dist_test", index=False)
    class_label_info.to_excel(writer, sheet_name="class_label_info", index=False)

    average_class_reports.to_excel(writer, sheet_name="avg_class_reports", index=False)
    per_fold_class_reports.to_excel(writer, sheet_name="per_fold_class_reports", index=False)
    test_class_reports.to_excel(writer, sheet_name="final_test_class_reports", index=False)

    for model_name, cm_df in cv_confusion_matrices.items():
        cm_df.to_excel(writer, sheet_name=safe_excel_sheet_name(f"CV_{model_name}_cm"))
        normalize_confusion_matrix_rows(cm_df).to_excel(
            writer,
            sheet_name=safe_excel_sheet_name(f"CV_{model_name}_cm_norm"),
        )

    for model_name, cm_df in test_confusion_matrices.items():
        cm_df.to_excel(writer, sheet_name=safe_excel_sheet_name(f"Test_{model_name}_cm"))
        normalize_confusion_matrix_rows(cm_df).to_excel(
            writer,
            sheet_name=safe_excel_sheet_name(f"Test_{model_name}_cm_norm"),
        )

    model_parameters.to_excel(writer, sheet_name="model_parameters", index=False)
    experiment_info.to_excel(writer, sheet_name="experiment_info", index=False)


 
#  Save figures
 
save_presentation_summary_table(
    summary_df=summary_metrics,
    output_path=FIGURES_DIR / "all_models_level2_presentation_summary_table.png",
)

save_metric_barplot(
    summary_df=summary_metrics,
    metric_col="f1_macro_mean",
    title="Level 2 Model Comparison: Macro F1 Mean Across 5-Fold CV",
    output_path=FIGURES_DIR / "all_models_level2_macro_f1_barplot.png",
)

save_metric_barplot(
    summary_df=summary_metrics,
    metric_col="f1_weighted_mean",
    title="Level 2 Model Comparison: Weighted F1 Mean Across 5-Fold CV",
    output_path=FIGURES_DIR / "all_models_level2_weighted_f1_barplot.png",
)

save_metric_barplot(
    summary_df=summary_metrics,
    metric_col="balanced_accuracy_mean",
    title="Level 2 Model Comparison: Balanced Accuracy Mean Across 5-Fold CV",
    output_path=FIGURES_DIR / "all_models_level2_balanced_accuracy_barplot.png",
)

for model_name, cm_df in cv_confusion_matrices.items():
    save_confusion_matrix_heatmap(
        cm_df=cm_df,
        title=f"{model_name} - summed CV confusion matrix",
        output_path=FIGURES_DIR / f"CV_{model_name}_confusion_heatmap.png",
        normalized=False,
    )
    save_confusion_matrix_heatmap(
        cm_df=cm_df,
        title=f"{model_name} - normalized summed CV confusion matrix (%)",
        output_path=FIGURES_DIR / f"CV_{model_name}_confusion_heatmap_normalized.png",
        normalized=True,
    )

for model_name, cm_df in test_confusion_matrices.items():
    save_confusion_matrix_heatmap(
        cm_df=cm_df,
        title=f"{model_name} - final test confusion matrix",
        output_path=FIGURES_DIR / f"Test_{model_name}_confusion_heatmap.png",
        normalized=False,
    )
    save_confusion_matrix_heatmap(
        cm_df=cm_df,
        title=f"{model_name} - normalized final test confusion matrix (%)",
        output_path=FIGURES_DIR / f"Test_{model_name}_confusion_heatmap_normalized.png",
        normalized=True,
    )


 
# Print final summary
 
print("\nCross-validation and final test evaluation finished successfully.")

print("\nMain CV summary sorted by Macro F1:")
print(summary_metrics[[
    "model_name",
    "algorithm",
    "class_weight",
    "accuracy_mean",
    "balanced_accuracy_mean",
    "f1_macro_mean",
    "f1_weighted_mean",
    "recall_macro_mean",
    "training_time_seconds_mean",
]])

print("\nFinal held-out test summary sorted by Macro F1:")
print(test_metrics[[
    "model_name",
    "algorithm",
    "class_weight",
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "recall_macro",
    "training_time_seconds",
    "prediction_time_seconds",
]])

print("\nSaved files:")
print(f"Excel workbook: {EXCEL_PATH}")
print(f"Metadata JSON:  {METADATA_PATH}")
print(f"Split indices:  {SPLIT_PATH}")
print(f"Figures folder: {FIGURES_DIR}")

print("\nRecommended model-selection metric: f1_macro_mean")
print("Use final_test_metrics as a held-out confirmation, not for choosing between models.")
