# 01_compare_LR.py

"""
Compare Logistic Regression models for PBMC68k cell type prediction.

This script compares:
1. LR_no_weight
2. LR_balanced

using stratified 5-fold cross-validation on the 80% development set.

It saves:
- One Excel workbook containing all relevant tables.
- One metadata JSON file.
- One split index file.
- Summary table, confusion matrix, macro f1 barplot figures.

Dataset:
pbmc68k_preprocessed_for_training.h5ad

Label:
cell_type_level_1
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

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
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


# ============================================================
# 1. Basic settings
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20
LABEL_COLUMN = "cell_type_level_1"

DATA_PATH = Path(
    r"C:\Users\ferid\Downloads\capstone_demo\outputs\pbmc68k_preprocessed_for_training.h5ad"
)

PROJECT_DIR = Path(r"C:\Users\ferid\Downloads\capstone_demo\model_development")

RESULTS_DIR = PROJECT_DIR / "results" / "LR_level1"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_PATH = RESULTS_DIR / "LR_level1_CV_results.xlsx"
METADATA_PATH = RESULTS_DIR / "LR_level1_metadata.json"
SPLIT_PATH = RESULTS_DIR / "LR_level1_split_indices.npz"


# ============================================================
# 2. Helper functions
# ============================================================

def make_lr_model(class_weight):
    """
    Create a Logistic Regression model.

    Parameters
    ----------
    class_weight:
        None or "balanced"
    """
    return LogisticRegression(
        max_iter=1000,
        solver="saga",
        penalty="l2",
        C=1.0,
        class_weight=class_weight,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate evaluation metrics for one validation fold.
    """
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

    return metrics


def create_label_distribution_table(labels, table_name):
    """
    Create a label distribution table with counts and percentages.
    """
    counts = pd.Series(labels).value_counts().sort_index()
    percentages = counts / counts.sum() * 100

    df = pd.DataFrame({
        "class_name": counts.index,
        "count": counts.values,
        "percentage": percentages.values,
    })

    df.insert(0, "set_name", table_name)
    return df


def create_summary_table(fold_metrics_df):
    """
    Create mean and standard deviation summary for each model.
    """
    excluded_cols = {
        "model_name",
        "algorithm",
        "class_weight",
        "fold",
        "n_train_fold",
        "n_validation_fold",
    }

    metric_columns = [
        col for col in fold_metrics_df.columns
        if col not in excluded_cols
    ]

    summary_rows = []

    for model_name in fold_metrics_df["model_name"].unique():
        model_df = fold_metrics_df[fold_metrics_df["model_name"] == model_name]

        row = {
            "model_name": model_name,
            "algorithm": "Logistic Regression",
            "label_column": LABEL_COLUMN,
            "class_weight": model_df["class_weight"].iloc[0],
            "n_features": N_FEATURES,
            "n_classes": N_CLASSES,
            "n_splits_cv": N_SPLITS,
            "development_set_size": N_DEV,
            "test_set_size": N_TEST,
        }

        for metric in metric_columns:
            row[f"{metric}_mean"] = model_df[metric].mean()
            row[f"{metric}_std"] = model_df[metric].std()

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    if "f1_macro_mean" in summary_df.columns:
        summary_df = summary_df.sort_values("f1_macro_mean", ascending=False)

    return summary_df


def average_classification_reports(report_df):
    """
    Average per-class precision, recall, and F1 across CV folds.
    """
    rows = []

    for model_name in report_df["model_name"].unique():
        model_part = report_df[report_df["model_name"] == model_name]

        for class_name in CLASS_NAMES:
            class_part = model_part[model_part["class_name"] == class_name]

            row = {
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
            }

            rows.append(row)

    return pd.DataFrame(rows)


def save_metadata(metadata, path):
    """
    Save experiment metadata as JSON.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def save_confusion_matrix_table(cm_df, title, output_path):
    """
    Save a confusion matrix as a readable table image instead of a heatmap.

    This is better for imbalanced datasets because one large class, such as T cell,
    can dominate the heatmap color scale and make other values hard to read.
    """
    table_df = cm_df.copy()

    # Add row/column labels for clarity
    table_df.index.name = "True label"
    table_df.columns.name = "Predicted label"

    n_rows, n_cols = table_df.shape

    # Figure size is adjusted for readability
    fig_width = max(10, n_cols * 1.6)
    fig_height = max(4.5, n_rows * 0.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        rowLabels=table_df.index,
        colLabels=table_df.columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    # Make header row and row labels easier to read
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)

        # Header row
        if row == 0:
            cell.set_text_props(weight="bold", fontsize=11)

        # Row labels
        if col == -1:
            cell.set_text_props(weight="bold", fontsize=11)

        # Matrix values
        if row > 0 and col >= 0:
            cell.set_text_props(fontsize=13)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def normalize_confusion_matrix_rows(cm_df):
    """
    Normalize a confusion matrix row-wise so that each row sums to 1.
    This shows, for each true class, the proportion predicted as each class.
    """
    cm_values = cm_df.values.astype(float)

    row_sums = cm_values.sum(axis=1, keepdims=True)

    # Avoid division by zero
    normalized_values = np.divide(
        cm_values,
        row_sums,
        out=np.zeros_like(cm_values, dtype=float),
        where=row_sums != 0
    )

    normalized_df = pd.DataFrame(
        normalized_values,
        index=cm_df.index,
        columns=cm_df.columns,
    )

    return normalized_df

def save_normalized_confusion_matrix_table(cm_df, title, output_path):
    """
    Save a normalized confusion matrix as a readable table image.
    Values are shown as percentages with 2 decimal places.
    """
    normalized_df = normalize_confusion_matrix_rows(cm_df)

    # Convert proportions to percentage strings
    display_df = normalized_df.copy().applymap(lambda x: f"{x * 100:.2f}%")

    display_df.index.name = "True label"
    display_df.columns.name = "Predicted label"

    n_rows, n_cols = display_df.shape

    fig_width = max(10, n_cols * 1.6)
    fig_height = max(4.5, n_rows * 0.8)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        rowLabels=display_df.index,
        colLabels=display_df.columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)

        if row == 0:
            cell.set_text_props(weight="bold", fontsize=11)

        if col == -1:
            cell.set_text_props(weight="bold", fontsize=11)

        if row > 0 and col >= 0:
            cell.set_text_props(fontsize=13)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_metric_barplot(summary_df, metric_col, title, output_path):
    """
    Save a barplot for a selected metric.
    """
    plot_df = summary_df[["model_name", metric_col]].copy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(plot_df["model_name"], plot_df[metric_col])

    ax.set_title(title)
    ax.set_ylabel(metric_col)
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)

    for i, value in enumerate(plot_df[metric_col]):
        ax.text(i, value, f"{value:.3f}", ha="center", va="bottom")

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_summary_table_image(summary_df, output_path):
    """
    Save a compact summary table as PNG for PowerPoint.
    """
    important_cols = [
        "model_name",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "f1_macro_mean",
        "f1_weighted_mean",
        "recall_macro_mean",
        "training_time_seconds_mean",
    ]

    existing_cols = [col for col in important_cols if col in summary_df.columns]
    table_df = summary_df[existing_cols].copy()

    rename_dict = {
        "model_name": "Model",
        "accuracy_mean": "Accuracy",
        "balanced_accuracy_mean": "Balanced Acc.",
        "f1_macro_mean": "Macro F1",
        "f1_weighted_mean": "Weighted F1",
        "recall_macro_mean": "Macro Recall",
        "training_time_seconds_mean": "Train Time (s)",
    }

    table_df = table_df.rename(columns=rename_dict)

    for col in table_df.columns:
        if col != "Model":
            table_df[col] = table_df[col].apply(
                lambda x: f"{x:.3f}" if isinstance(x, (int, float, np.floating)) else x
            )

    fig, ax = plt.subplots(figsize=(11, 2.8))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    ax.set_title("Logistic Regression Level 1 CV Summary", pad=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# ============================================================
# 3. Load dataset
# ============================================================

print("Loading dataset...")
adata = sc.read_h5ad(DATA_PATH)

print("Dataset loaded:")
print(adata)

if LABEL_COLUMN not in adata.obs.columns:
    raise ValueError(f"Label column '{LABEL_COLUMN}' was not found in adata.obs.")

X = adata.X
y_text = adata.obs[LABEL_COLUMN].astype(str).values

if sparse.issparse(X):
    X = X.tocsr()

N_CELLS = adata.n_obs
N_FEATURES = adata.n_vars

print(f"\nNumber of cells: {N_CELLS}")
print(f"Number of features/HVGs: {N_FEATURES}")
print(f"Label column: {LABEL_COLUMN}")


# ============================================================
# 4. Encode labels
# ============================================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

CLASS_NAMES = list(label_encoder.classes_)
N_CLASSES = len(CLASS_NAMES)

print("\nEncoded classes:")
for encoded_label, class_name in enumerate(CLASS_NAMES):
    print(f"{encoded_label}: {class_name}")


# ============================================================
# 5. Create 80/20 development-test split
# ============================================================

all_indices = np.arange(N_CELLS)

dev_indices, test_indices = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

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

print(f"\nSaved split indices to: {SPLIT_PATH}")
print(f"Development set size: {N_DEV}")
print(f"Test set size: {N_TEST}")


# ============================================================
# 6. Prepare label distribution and class info tables
# ============================================================

label_distribution_total = create_label_distribution_table(
    labels=y_text,
    table_name="total_dataset",
)

label_distribution_training_set = create_label_distribution_table(
    labels=y_dev_text,
    table_name="development_set_used_for_cv",
)

label_distribution_test_set = create_label_distribution_table(
    labels=y_test_text,
    table_name="final_test_set",
)

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


# ============================================================
# 7. Define candidate models
# ============================================================

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
}


model_parameters_rows = []

for model_name, model_info in models.items():
    params = model_info["model"].get_params()

    row = {
        "model_name": model_name,
        "algorithm": model_info["algorithm"],
        "class_weight": str(params.get("class_weight")),
        "solver": params.get("solver"),
        "penalty": params.get("penalty"),
        "C": params.get("C"),
        "max_iter": params.get("max_iter"),
        "random_state": params.get("random_state"),
        "n_jobs": params.get("n_jobs"),
    }

    model_parameters_rows.append(row)

model_parameters = pd.DataFrame(model_parameters_rows)


# ============================================================
# 8. Cross-validation
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

fold_metrics_rows = []
per_fold_report_rows = []
confusion_matrices = {}

print("\nStarting 5-fold stratified cross-validation...")

for model_name, model_info in models.items():
    model = model_info["model"]
    algorithm = model_info["algorithm"]
    class_weight = model_info["class_weight"]

    print(f"\nModel: {model_name}")

    model_fold_cms = []

    for fold_number, (train_idx, val_idx) in enumerate(cv.split(X_dev, y_dev), start=1):
        print(f"  Fold {fold_number}/{N_SPLITS}")

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

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_val_fold)
        else:
            y_proba = None

        metrics = calculate_metrics(
            y_true=y_val_fold,
            y_pred=y_pred,
            y_proba=y_proba,
        )

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

        cm = confusion_matrix(
            y_val_fold,
            y_pred,
            labels=np.arange(N_CLASSES),
        )

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
                "fold": fold_number,
                "class_name": class_name,
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1-score": report[class_name]["f1-score"],
                "support": report[class_name]["support"],
            })

    summed_cm = np.sum(model_fold_cms, axis=0)

    confusion_matrices[model_name] = pd.DataFrame(
        summed_cm,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )


# ============================================================
# 9. Create final result tables
# ============================================================

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

summary_metrics = create_summary_table(fold_metrics)

per_fold_class_reports = pd.DataFrame(per_fold_report_rows)
average_class_reports = average_classification_reports(per_fold_class_reports)

LR_no_weight_report = average_class_reports[
    average_class_reports["model_name"] == "LR_no_weight"
].reset_index(drop=True)

LR_balanced_report = average_class_reports[
    average_class_reports["model_name"] == "LR_balanced"
].reset_index(drop=True)

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
        "cv_type",
        "n_splits_cv",
        "random_state",
        "model_family",
        "models_compared",
        "model_selection_metric_recommended",
    ],
    "value": [
        "LR_level1_CV",
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
        "StratifiedKFold",
        N_SPLITS,
        RANDOM_STATE,
        "Logistic Regression",
        ", ".join(models.keys()),
        "f1_macro_mean",
    ],
})


# ============================================================
# 10. Save metadata JSON
# ============================================================

metadata = {
    "experiment_name": "LR_level1_CV",
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
    "cv_type": "StratifiedKFold",
    "n_splits_cv": N_SPLITS,
    "random_state": RANDOM_STATE,
    "model_family": "Logistic Regression",
    "models_compared": list(models.keys()),
    "model_parameters": model_parameters.to_dict(orient="records"),
    "model_selection_metric_recommended": "f1_macro_mean",
}

save_metadata(metadata, METADATA_PATH)


# ============================================================
# 11. Save one Excel workbook with all tables
# ============================================================

with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
    summary_metrics.to_excel(writer, sheet_name="summary_metrics", index=False)
    fold_metrics.to_excel(writer, sheet_name="fold_metrics", index=False)

    label_distribution_total.to_excel(
        writer,
        sheet_name="label_distribution_total",
        index=False,
    )

    label_distribution_training_set.to_excel(
        writer,
        sheet_name="label_distribution_training_set",
        index=False,
    )

    label_distribution_test_set.to_excel(
        writer,
        sheet_name="label_distribution_test_set",
        index=False,
    )

    class_label_info.to_excel(writer, sheet_name="class_label_info", index=False)

    confusion_matrices["LR_no_weight"].to_excel(
        writer,
        sheet_name="LR_no_weight_confusion",
    )

    confusion_matrices["LR_balanced"].to_excel(
        writer,
        sheet_name="LR_balanced_confusion",
    )

    LR_no_weight_report.to_excel(
        writer,
        sheet_name="LR_no_weight_report",
        index=False,
    )

    LR_balanced_report.to_excel(
        writer,
        sheet_name="LR_balanced_report",
        index=False,
    )

    per_fold_class_reports.to_excel(
        writer,
        sheet_name="per_fold_class_reports",
        index=False,
    )

    LR_no_weight_confusion_normalized = normalize_confusion_matrix_rows(
        confusion_matrices["LR_no_weight"]
    )

    LR_balanced_confusion_normalized = normalize_confusion_matrix_rows(
        confusion_matrices["LR_balanced"]
    )

    LR_no_weight_confusion_normalized.to_excel(
        writer,
        sheet_name="LR_no_weight_conf_norm",
    )

    LR_balanced_confusion_normalized.to_excel(
        writer,
        sheet_name="LR_balanced_conf_norm",
    )

    model_parameters.to_excel(writer, sheet_name="model_parameters", index=False)
    experiment_info.to_excel(writer, sheet_name="experiment_info", index=False)


# ============================================================
# 12. Save figures
# ============================================================

save_summary_table_image(
    summary_df=summary_metrics,
    output_path=FIGURES_DIR / "LR_level1_summary_table.png",
)

save_metric_barplot(
    summary_df=summary_metrics,
    metric_col="f1_macro_mean",
    title="Logistic Regression Level 1: Macro F1 Comparison",
    output_path=FIGURES_DIR / "LR_level1_macro_f1_barplot.png",
)

save_metric_barplot(
    summary_df=summary_metrics,
    metric_col="balanced_accuracy_mean",
    title="Logistic Regression Level 1: Balanced Accuracy Comparison",
    output_path=FIGURES_DIR / "LR_level1_balanced_accuracy_barplot.png",
)

save_confusion_matrix_table(
    cm_df=confusion_matrices["LR_no_weight"],
    title="LR no weight - summed CV confusion matrix",
    output_path=FIGURES_DIR / "LR_level1_no_weight_confusion_table.png",
)

save_confusion_matrix_table(
    cm_df=confusion_matrices["LR_balanced"],
    title="LR balanced - summed CV confusion matrix",
    output_path=FIGURES_DIR / "LR_level1_balanced_confusion_table.png",
)

save_normalized_confusion_matrix_table(
    cm_df=confusion_matrices["LR_no_weight"],
    title="LR no weight - normalized summed CV confusion matrix (%)",
    output_path=FIGURES_DIR / "LR_level1_no_weight_confusion_table_normalized.png",
)

save_normalized_confusion_matrix_table(
    cm_df=confusion_matrices["LR_balanced"],
    title="LR balanced - normalized summed CV confusion matrix (%)",
    output_path=FIGURES_DIR / "LR_level1_balanced_confusion_table_normalized.png",
)


# ============================================================
# 13. Print final summary
# ============================================================

print("\nCross-validation finished successfully.")

print("\nMain summary:")
print(summary_metrics[[
    "model_name",
    "accuracy_mean",
    "balanced_accuracy_mean",
    "f1_macro_mean",
    "f1_weighted_mean",
    "recall_macro_mean",
    "training_time_seconds_mean",
]])

print("\nSaved files:")
print(f"Excel workbook: {EXCEL_PATH}")
print(f"Metadata JSON:  {METADATA_PATH}")
print(f"Split indices:  {SPLIT_PATH}")
print(f"Figures folder: {FIGURES_DIR}")

print("\nRecommended model-selection metric: f1_macro_mean")