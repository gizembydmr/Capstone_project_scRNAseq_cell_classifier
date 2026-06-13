# 02_train_final_LR_balanced_model_level3.py

"""
Train the final pancreas Logistic Regression model for Level 3 labels.

Selected model:
- Logistic Regression
- balanced class weight
- label: cell_type_level_3
- input: preprocessed pancreas AnnData with 2000 HVGs

The model is trained on the saved 80% development set and evaluated once on
its untouched 20% test set.
"""

from pathlib import Path
import json
import time
from datetime import datetime
import warnings

import joblib
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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


# Settings
RANDOM_STATE = 42
LABEL_COLUMN = "cell_type_level_3"

DATA_PATH = Path(
    r"C:\Users\ferid\Downloads\capstone_demo\pancreas_model\03_preprocess_train_pancreas_outputs\Tabula_Sapiens_Pancreas_preprocessed_for_training.h5ad"
)

PROJECT_DIR = Path(
    r"C:\Users\ferid\Downloads\capstone_demo\pancreas_model\pancreas_model_development"
)

SPLIT_PATH = (
    PROJECT_DIR
    / "results"
    / "all_models_level3"
    / "pancreas_all_models_level3_split_indices.npz"
)

RESULTS_DIR = PROJECT_DIR / "results" / "final_LR_balanced_level3"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_DIR = PROJECT_DIR / "models" / "LR_balanced_level3"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "pancreas_LR_balanced_level3_final"
MODEL_PATH = MODEL_DIR / "pancreas_LR_balanced_level3_final_model_bundle.joblib"
METADATA_PATH = MODEL_DIR / "pancreas_LR_balanced_level3_final_model_metadata.json"
EXCEL_PATH = RESULTS_DIR / "pancreas_LR_balanced_level3_final_test_results.xlsx"


def make_final_lr_model():
    return LogisticRegression(
        max_iter=1000,
        solver="saga",
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


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


def normalize_confusion_matrix_rows(cm_df):
    values = cm_df.values.astype(float)
    row_sums = values.sum(axis=1, keepdims=True)

    normalized = np.divide(
        values,
        row_sums,
        out=np.zeros_like(values, dtype=float),
        where=row_sums != 0,
    )

    return pd.DataFrame(normalized, index=cm_df.index, columns=cm_df.columns)


def save_confusion_matrix_table(cm_df, title, output_path):
    table_df = cm_df.copy()
    table_df.index.name = "True label"
    table_df.columns.name = "Predicted label"

    fig_width = max(10, len(cm_df.columns) * 1.25)
    fig_height = max(5, len(cm_df.index) * 0.85)
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
    table.set_fontsize(10)
    table.scale(1.2, 1.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        if row == 0 or col == -1:
            cell.set_text_props(weight="bold", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_normalized_confusion_matrix_table(cm_df, title, output_path):
    normalized_df = normalize_confusion_matrix_rows(cm_df)
    display_df = normalized_df.apply(lambda col: col.map(lambda x: f"{x * 100:.2f}%"))

    fig_width = max(10, len(cm_df.columns) * 1.25)
    fig_height = max(5, len(cm_df.index) * 0.85)
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
    table.set_fontsize(10)
    table.scale(1.2, 1.7)

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        if row == 0 or col == -1:
            cell.set_text_props(weight="bold", fontsize=9)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_normalized_confusion_matrix_heatmap(cm_df, title, output_path):
    normalized_df = normalize_confusion_matrix_rows(cm_df)
    values = normalized_df.values * 100

    light_cmap = LinearSegmentedColormap.from_list(
        "light_blue",
        ["#ffffff", "#d8ecff", "#a9d4ff"],
    )

    fig_width = max(9.5, len(normalized_df.columns) * 1.15)
    fig_height = max(6, len(normalized_df.index) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(values, cmap=light_cmap, vmin=0, vmax=100)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(normalized_df.columns)))
    ax.set_yticks(np.arange(len(normalized_df.index)))
    ax.set_xticklabels(normalized_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(normalized_df.index)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}%", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-wise percentage (%)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_label_distribution_table(labels, table_name):
    counts = pd.Series(labels).value_counts().sort_index()
    percentages = counts / counts.sum() * 100

    return pd.DataFrame({
        "set_name": table_name,
        "class_name": counts.index,
        "count": counts.values,
        "percentage": percentages.values,
    })


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# Load data
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


# Training preprocessing metadata
training_preprocessing = adata.uns.get("preprocessing", {})
training_target_sum = training_preprocessing.get("target_sum", None)
training_min_counts = training_preprocessing.get("min_counts", None)
training_min_genes = training_preprocessing.get("min_genes", None)

if training_target_sum is None:
    raise ValueError(
        "target_sum was not found in adata.uns['preprocessing']. "
        "Please check the preprocessed training dataset."
    )

print("\nTraining preprocessing settings:")
print(f"target_sum: {training_target_sum}")
print(f"min_counts: {training_min_counts}")
print(f"min_genes: {training_min_genes}")


# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

CLASS_NAMES = list(label_encoder.classes_)
N_CLASSES = len(CLASS_NAMES)

print("\nEncoded classes:")
for encoded_label, class_name in enumerate(CLASS_NAMES):
    print(f"{encoded_label}: {class_name}")


# Load saved split
if not SPLIT_PATH.exists():
    raise FileNotFoundError(
        f"Split file not found:\n{SPLIT_PATH}\n"
        "Please run 01_compare_all_models_level3_pancreas.py first."
    )

split_data = np.load(SPLIT_PATH)
dev_indices = split_data["dev_indices"]
test_indices = split_data["test_indices"]

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

print(f"\nDevelopment set size: {N_DEV}")
print(f"Test set size: {N_TEST}")


# Train final model
model = make_final_lr_model()

print("\nTraining final balanced Logistic Regression model on full development set...")
start_time = time.time()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_dev, y_dev)

training_time_seconds = time.time() - start_time

print(
    f"Training finished in {training_time_seconds:.2f} seconds "
    f"({training_time_seconds / 60:.2f} minutes)."
)


# Evaluate final model
test_start_time = time.time()
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
prediction_time_seconds = time.time() - test_start_time

metrics = calculate_metrics(y_test, y_pred, y_proba)

metrics_row = {
    "model_name": MODEL_NAME,
    "algorithm": "Logistic Regression",
    "label_column": LABEL_COLUMN,
    "class_weight": "balanced",
    "n_features": N_FEATURES,
    "n_classes": N_CLASSES,
    "development_set_size": N_DEV,
    "test_set_size": N_TEST,
    "training_time_seconds": training_time_seconds,
    "prediction_time_seconds": prediction_time_seconds,
    **metrics,
}

metrics_df = pd.DataFrame([metrics_row])

classification_report_dict = classification_report(
    y_test,
    y_pred,
    labels=np.arange(N_CLASSES),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0,
)
classification_report_df = pd.DataFrame(classification_report_dict).transpose()

cm = confusion_matrix(y_test, y_pred, labels=np.arange(N_CLASSES))
cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
cm_norm_df = normalize_confusion_matrix_rows(cm_df)

test_predictions = pd.DataFrame({
    "cell_index": test_indices,
    "barcode": adata.obs_names[test_indices],
    "true_label": y_test_text,
    "predicted_label": label_encoder.inverse_transform(y_pred),
})

for i, class_name in enumerate(CLASS_NAMES):
    test_predictions[f"probability_{class_name}"] = y_proba[:, i]


# Gene and label info
feature_info = pd.DataFrame({
    "feature_order": np.arange(N_FEATURES),
    "feature_id": list(adata.var_names),
})

if "gene_symbol" in adata.var.columns:
    feature_info["gene_symbol"] = adata.var["gene_symbol"].astype(str).values

label_info = pd.DataFrame({
    "encoded_label": np.arange(N_CLASSES),
    "class_name": CLASS_NAMES,
})


# Save model bundle
model_bundle = {
    "model": model,
    "label_encoder": label_encoder,
    "model_name": MODEL_NAME,
    "model_type": "LogisticRegression",
    "label_column": LABEL_COLUMN,
    "class_names": CLASS_NAMES,
    "training_gene_order": list(adata.var_names),
    "feature_names": list(adata.var_names),
    "gene_symbols": (
        list(adata.var["gene_symbol"].astype(str).values)
        if "gene_symbol" in adata.var.columns
        else None
    ),
    "n_features": int(N_FEATURES),
    "preprocessing": {
        "target_sum": float(training_target_sum),
        "min_counts": int(training_min_counts) if training_min_counts is not None else None,
        "min_genes": int(training_min_genes) if training_min_genes is not None else None,
        "normalization": "library_size_normalization",
        "log_transform": "log1p",
        "gene_id_type": "Ensembl ID",
        "feature_selection": "highly_variable_genes",
        "n_hvgs": int(N_FEATURES),
    },
    "inference_note": (
        "Uploaded/query data must use the saved training target_sum and must be "
        "aligned to training_gene_order before prediction."
    ),
}

joblib.dump(model_bundle, MODEL_PATH)
print(f"\nSaved final model bundle to:\n{MODEL_PATH}")


# Save metadata
metadata = {
    "model_name": MODEL_NAME,
    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_path": str(DATA_PATH),
    "split_path": str(SPLIT_PATH),
    "label_column": LABEL_COLUMN,
    "n_cells_total": int(N_CELLS),
    "n_features": int(N_FEATURES),
    "n_classes": int(N_CLASSES),
    "class_names": CLASS_NAMES,
    "development_set_size": int(N_DEV),
    "test_set_size": int(N_TEST),
    "model_type": "LogisticRegression",
    "model_parameters": model.get_params(),
    "model_bundle_path": str(MODEL_PATH),
    "preprocessing": model_bundle["preprocessing"],
    "training_time_seconds": training_time_seconds,
    "prediction_time_seconds": prediction_time_seconds,
    "test_metrics": metrics,
    "inference_requirement": (
        "Uploaded/query data must be preprocessed using the saved training target_sum, "
        "then aligned to the same HVG feature space: same Ensembl IDs, same HVG list, "
        "same gene order, and same log1p transformation before prediction."
    ),
}

save_json(metadata, METADATA_PATH)
print(f"Saved model metadata to:\n{METADATA_PATH}")


# Save workbook
label_distribution_total = create_label_distribution_table(y_text, "total_dataset")
label_distribution_development = create_label_distribution_table(
    y_dev_text,
    "development_set_used_for_final_training",
)
label_distribution_test = create_label_distribution_table(y_test_text, "final_test_set")

experiment_info = pd.DataFrame({
    "field": [
        "model_name",
        "date_time",
        "dataset_path",
        "split_path",
        "label_column",
        "n_cells_total",
        "n_features",
        "n_classes",
        "development_set_size",
        "test_set_size",
        "model_type",
        "class_weight",
        "training_target_sum",
        "training_min_counts",
        "training_min_genes",
        "normalization",
        "log_transform",
        "gene_id_type",
        "feature_selection",
        "n_hvgs",
        "training_time_seconds",
        "prediction_time_seconds",
        "model_bundle_path",
    ],
    "value": [
        MODEL_NAME,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        str(DATA_PATH),
        str(SPLIT_PATH),
        LABEL_COLUMN,
        N_CELLS,
        N_FEATURES,
        N_CLASSES,
        N_DEV,
        N_TEST,
        "LogisticRegression",
        "balanced",
        training_target_sum,
        training_min_counts,
        training_min_genes,
        "library_size_normalization",
        "log1p",
        "Ensembl ID",
        "highly_variable_genes",
        N_FEATURES,
        training_time_seconds,
        prediction_time_seconds,
        str(MODEL_PATH),
    ],
})

with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
    metrics_df.to_excel(writer, sheet_name="test_metrics", index=False)
    classification_report_df.to_excel(writer, sheet_name="classification_report")
    cm_df.to_excel(writer, sheet_name="confusion_matrix_raw")
    cm_norm_df.to_excel(writer, sheet_name="confusion_matrix_normalized")
    test_predictions.to_excel(writer, sheet_name="test_predictions", index=False)
    label_distribution_total.to_excel(writer, sheet_name="label_distribution_total", index=False)
    label_distribution_development.to_excel(
        writer,
        sheet_name="label_distribution_development",
        index=False,
    )
    label_distribution_test.to_excel(writer, sheet_name="label_distribution_test", index=False)
    feature_info.to_excel(writer, sheet_name="feature_gene_order", index=False)
    label_info.to_excel(writer, sheet_name="label_encoder_classes", index=False)
    experiment_info.to_excel(writer, sheet_name="experiment_info", index=False)

print(f"Saved final test results workbook to:\n{EXCEL_PATH}")


# Save figures
save_confusion_matrix_table(
    cm_df=cm_df,
    title="Final pancreas LR balanced - test confusion matrix",
    output_path=FIGURES_DIR / "pancreas_LR_balanced_level3_final_test_confusion_table.png",
)

save_normalized_confusion_matrix_table(
    cm_df=cm_df,
    title="Final pancreas LR balanced - normalized test confusion matrix (%)",
    output_path=FIGURES_DIR / "pancreas_LR_balanced_level3_final_test_confusion_table_normalized.png",
)

save_normalized_confusion_matrix_heatmap(
    cm_df=cm_df,
    title="Final pancreas LR balanced - normalized test confusion matrix (%)",
    output_path=FIGURES_DIR / "pancreas_LR_balanced_level3_final_test_confusion_heatmap_light.png",
)

print(f"Saved figures to:\n{FIGURES_DIR}")


# Summary
print("\nFinal test evaluation completed.")
print("\nMain test metrics:")
print(metrics_df[[
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "matthews_corrcoef",
    "cohen_kappa",
    "log_loss",
]])

print("\nImportant saved files:")
print(f"Model bundle: {MODEL_PATH}")
print(f"Metadata:     {METADATA_PATH}")
print(f"Results:      {EXCEL_PATH}")
print(f"Figures:      {FIGURES_DIR}")
