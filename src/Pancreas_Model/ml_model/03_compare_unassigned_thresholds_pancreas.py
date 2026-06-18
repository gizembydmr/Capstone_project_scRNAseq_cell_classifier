# 03_compare_unassigned_thresholds_pancreas.py

"""
Compare confidence thresholds for adding an Unassigned label to the final
pancreas Level 3 Logistic Regression model.

This script trains the same balanced LR model on the saved 80% development set and evaluates
confidence thresholds on the untouched 20% test set.

It saves:
1. Threshold comparison Excel workbook
2. Test predictions with confidence scores
3. Confusion matrices for each threshold
4. Presentation-ready summary table figure
5. Threshold trade-off line plot
6. Assigned vs unassigned stacked bar plot
7. Summary metadata JSON
"""

from pathlib import Path
import json
import time
from datetime import datetime
import warnings

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


RANDOM_STATE = 42
LABEL_COLUMN = "cell_type_level_3"
UNASSIGNED_LABEL = "Unassigned"
SELECTED_THRESHOLD = 0.85
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

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

RESULTS_DIR = PROJECT_DIR / "results" / "unassigned_thresholds_level3_pancreas"
FIGURES_DIR = RESULTS_DIR / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_PATH = RESULTS_DIR / "pancreas_LR_balanced_level3_unassigned_threshold_comparison.xlsx"
METADATA_PATH = RESULTS_DIR / "pancreas_LR_balanced_level3_unassigned_threshold_comparison_metadata.json"


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


def save_normalized_confusion_matrix_light_heatmap(cm_df, title, output_path):
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

    ax.set_title(title, fontsize=15, fontweight="bold", pad=18)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    ax.set_xticks(np.arange(len(normalized_df.columns)))
    ax.set_yticks(np.arange(len(normalized_df.index)))

    ax.set_xticklabels(normalized_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(normalized_df.index)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:.2f}%",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Row-wise percentage (%)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()



def create_presentation_summary(threshold_summary_df):
    summary = threshold_summary_df[[
        "threshold",
        "assigned_percentage",
        "unassigned_percentage",
        "assigned_accuracy",
        "assigned_f1_macro",
        "strict_accuracy_with_unassigned_as_wrong",
    ]].copy()

    summary = summary.rename(columns={
        "threshold": "Threshold",
        "assigned_percentage": "Assigned cells (%)",
        "unassigned_percentage": "Unassigned cells (%)",
        "assigned_accuracy": "Accuracy on assigned cells (%)",
        "assigned_f1_macro": "Macro F1 on assigned cells (%)",
        "strict_accuracy_with_unassigned_as_wrong": "Strict accuracy (%)",
    })

    percent_columns = [
        "Accuracy on assigned cells (%)",
        "Macro F1 on assigned cells (%)",
        "Strict accuracy (%)",
    ]

    for col in percent_columns:
        summary[col] = summary[col] * 100

    return summary


def save_threshold_tradeoff_plot(threshold_summary_df, output_path):
    plot_df = threshold_summary_df.copy()

    plt.figure(figsize=(11, 6.5))

    plt.plot(
        plot_df["threshold"],
        plot_df["assigned_percentage"],
        marker="o",
        linewidth=2.5,
        label="Assigned cells (%)",
    )
    plt.plot(
        plot_df["threshold"],
        plot_df["unassigned_percentage"],
        marker="o",
        linewidth=2.5,
        label="Unassigned cells (%)",
    )
    plt.plot(
        plot_df["threshold"],
        plot_df["assigned_accuracy"] * 100,
        marker="s",
        linewidth=2.5,
        label="Accuracy on assigned cells (%)",
    )
    plt.plot(
        plot_df["threshold"],
        plot_df["assigned_f1_macro"] * 100,
        marker="s",
        linewidth=2.5,
        label="Macro F1 on assigned cells (%)",
    )

    plt.title(
        "Confidence Threshold Trade-off for Pancreas Level 3 Balanced LR Model",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    plt.xlabel("Confidence threshold", fontsize=12)
    plt.ylabel("Percentage / score (%)", fontsize=12)
    plt.ylim(0, 105)
    plt.xticks(plot_df["threshold"])
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="best", frameon=True)

    plt.figtext(
        0.01,
        0.01,
        "Higher thresholds reject more uncertain cells. Assigned-cell metrics are calculated only for cells not labeled Unassigned.",
        ha="left",
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_assigned_unassigned_bar_plot(threshold_summary_df, output_path):
    plot_df = threshold_summary_df.copy()
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(11, 6.5))

    ax.bar(
        x,
        plot_df["assigned_percentage"],
        label="Assigned",
    )
    ax.bar(
        x,
        plot_df["unassigned_percentage"],
        bottom=plot_df["assigned_percentage"],
        label="Unassigned",
    )

    ax.set_title(
        "Assigned and Unassigned Cells by Confidence Threshold",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Confidence threshold", fontsize=12)
    ax.set_ylabel("Cells (%)", fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}" for t in plot_df["threshold"]])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=True,
    )
    ax.grid(True, axis="y", alpha=0.25)

    for i, row in plot_df.iterrows():
        assigned = row["assigned_percentage"]
        unassigned = row["unassigned_percentage"]

        if assigned >= 8:
            ax.text(
                i,
                assigned / 2,
                f"{assigned:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

        if unassigned >= 8:
            ax.text(
                i,
                assigned + unassigned / 2,
                f"{unassigned:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_threshold_summary_table(presentation_summary_df, output_path, selected_threshold=None):
    display_df = presentation_summary_df.copy()

    display_df["Threshold"] = display_df["Threshold"].map(lambda x: f"{x:.2f}")

    for col in display_df.columns:
        if col != "Threshold":
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")

    fig_width = 15
    fig_height = max(4.8, 0.55 * len(display_df) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)

    header_color = "#1f4e79"
    row_light = "#ffffff"
    row_dark = "#eef5fb"
    selected_color = "#fff2cc"

    selected_table_row = None
    if selected_threshold is not None:
        selected_rows = presentation_summary_df.index[
            np.isclose(presentation_summary_df["Threshold"], selected_threshold)
        ].tolist()
        if selected_rows:
            selected_table_row = int(selected_rows[0]) + 1

    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.6)

        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="white", weight="bold", fontsize=9)
        else:
            if selected_table_row is not None and row == selected_table_row:
                cell.set_facecolor(selected_color)
                cell.set_text_props(weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor(row_dark)
            else:
                cell.set_facecolor(row_light)

    ax.set_title(
        "Pancreas Level 3 Balanced LR Confidence Threshold Comparison",
        fontsize=17,
        fontweight="bold",
        pad=18,
    )

    ax.text(
        0.5,
        -0.08,
        "The 0.85 threshold is highlighted as the selected pancreas threshold; threshold choice should consider both coverage and reliability.",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def save_recommended_threshold_note(threshold_summary_df, output_path, max_unassigned_percentage=20):
    candidate_df = threshold_summary_df[
        threshold_summary_df["unassigned_percentage"] <= max_unassigned_percentage
    ].copy()

    if candidate_df.empty:
        chosen_row = threshold_summary_df.loc[
            threshold_summary_df["assigned_f1_macro"].idxmax()
        ]
        reason = (
            "No tested threshold kept the unassigned percentage under "
            f"{max_unassigned_percentage}%. The shown threshold has the highest assigned-cell macro F1."
        )
    else:
        chosen_row = candidate_df.loc[candidate_df["assigned_f1_macro"].idxmax()]
        reason = (
            "This threshold has the highest assigned-cell macro F1 among thresholds with "
            f"unassigned cells <= {max_unassigned_percentage}%."
        )

    note = {
        "suggested_threshold": float(chosen_row["threshold"]),
        "assigned_percentage": float(chosen_row["assigned_percentage"]),
        "unassigned_percentage": float(chosen_row["unassigned_percentage"]),
        "assigned_accuracy": float(chosen_row["assigned_accuracy"]),
        "assigned_f1_macro": float(chosen_row["assigned_f1_macro"]),
        "selection_rule": reason,
    }

    save_json(note, output_path)
    return note


def apply_unassigned_threshold(predicted_labels, confidence_scores, threshold):
    final_labels = predicted_labels.copy().astype(object)
    final_labels[confidence_scores < threshold] = UNASSIGNED_LABEL
    return final_labels


def evaluate_threshold(y_true_text, predicted_labels, confidence_scores, threshold):
    final_labels = apply_unassigned_threshold(
        predicted_labels,
        confidence_scores,
        threshold,
    )

    assigned_mask = final_labels != UNASSIGNED_LABEL
    n_total = len(final_labels)
    n_assigned = int(assigned_mask.sum())
    n_unassigned = int(n_total - n_assigned)

    row = {
        "threshold": threshold,
        "n_total_cells": n_total,
        "n_assigned": n_assigned,
        "n_unassigned": n_unassigned,
        "assigned_percentage": n_assigned / n_total * 100,
        "unassigned_percentage": n_unassigned / n_total * 100,
        "mean_confidence_all_cells": float(np.mean(confidence_scores)),
        "median_confidence_all_cells": float(np.median(confidence_scores)),
    }

    row["strict_accuracy_with_unassigned_as_wrong"] = float(
        np.mean(final_labels == y_true_text)
    )

    if n_assigned > 0:
        y_true_assigned = y_true_text[assigned_mask]
        y_pred_assigned = final_labels[assigned_mask]

        row.update({
            "assigned_accuracy": accuracy_score(y_true_assigned, y_pred_assigned),
            "assigned_balanced_accuracy": balanced_accuracy_score(y_true_assigned, y_pred_assigned),
            "assigned_f1_macro": f1_score(y_true_assigned, y_pred_assigned, average="macro", zero_division=0),
            "assigned_f1_weighted": f1_score(y_true_assigned, y_pred_assigned, average="weighted", zero_division=0),
            "assigned_precision_macro": precision_score(y_true_assigned, y_pred_assigned, average="macro", zero_division=0),
            "assigned_recall_macro": recall_score(y_true_assigned, y_pred_assigned, average="macro", zero_division=0),
        })
    else:
        row.update({
            "assigned_accuracy": np.nan,
            "assigned_balanced_accuracy": np.nan,
            "assigned_f1_macro": np.nan,
            "assigned_f1_weighted": np.nan,
            "assigned_precision_macro": np.nan,
            "assigned_recall_macro": np.nan,
        })

    return row, final_labels


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

training_preprocessing = adata.uns.get("preprocessing", {})
training_target_sum = training_preprocessing.get("target_sum", None)
training_min_counts = training_preprocessing.get("min_counts", None)
training_min_genes = training_preprocessing.get("min_genes", None)

if training_target_sum is None:
    raise ValueError(
        "target_sum was not found in adata.uns['preprocessing']. "
        "Please check the preprocessed training AnnData file."
    )

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

CLASS_NAMES = list(label_encoder.classes_)
N_CLASSES = len(CLASS_NAMES)
PREDICTION_LABELS_WITH_UNASSIGNED = CLASS_NAMES + [UNASSIGNED_LABEL]

print("\nEncoded classes:")
for encoded_label, class_name in enumerate(CLASS_NAMES):
    print(f"{encoded_label}: {class_name}")

if not SPLIT_PATH.exists():
    raise FileNotFoundError(
        f"Split file not found:\n{SPLIT_PATH}\n"
        "Please run 01_compare_all_models_level3_pancreas.py first."
    )

split_data = np.load(SPLIT_PATH)
dev_indices = split_data["dev_indices"]
test_indices = split_data["test_indices"]

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

model = make_final_lr_model()

print("\nTraining Logistic Regression model on full development set...")
start_time = time.time()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_dev, y_dev)

training_time_seconds = time.time() - start_time
print(f"Training finished in {training_time_seconds:.2f} seconds.")

print("\nPredicting probabilities on untouched test set...")
prediction_start_time = time.time()

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

prediction_time_seconds = time.time() - prediction_start_time

predicted_labels = label_encoder.inverse_transform(y_pred)
confidence_scores = np.max(y_proba, axis=1)

baseline_metrics = calculate_metrics(y_test, y_pred, y_proba)
baseline_metrics_df = pd.DataFrame([{
    "model_name": "pancreas_LR_balanced_level3_final_without_unassigned",
    "algorithm": "Logistic Regression",
    "label_column": LABEL_COLUMN,
    "class_weight": "balanced",
    "n_features": N_FEATURES,
    "n_classes": N_CLASSES,
    "development_set_size": N_DEV,
    "test_set_size": N_TEST,
    "training_time_seconds": training_time_seconds,
    "prediction_time_seconds": prediction_time_seconds,
    **baseline_metrics,
}])

threshold_rows = []
threshold_prediction_tables = {}
threshold_class_reports = {}
threshold_confusion_matrices = {}
threshold_confusion_matrices_norm = {}

base_predictions = pd.DataFrame({
    "cell_index": test_indices,
    "barcode": adata.obs_names[test_indices],
    "true_label": y_test_text,
    "predicted_label_before_threshold": predicted_labels,
    "confidence_score": confidence_scores,
})

for i, class_name in enumerate(CLASS_NAMES):
    base_predictions[f"probability_{class_name}"] = y_proba[:, i]

for threshold in THRESHOLDS:
    print(f"Evaluating threshold {threshold:.2f}...")

    row, final_labels = evaluate_threshold(
        y_test_text,
        predicted_labels,
        confidence_scores,
        threshold,
    )

    threshold_rows.append(row)

    threshold_name = str(threshold).replace(".", "_")
    sheet_suffix = threshold_name.replace("0_", "")

    prediction_df = base_predictions.copy()
    prediction_df["threshold"] = threshold
    prediction_df["final_label_after_threshold"] = final_labels
    prediction_df["is_unassigned"] = final_labels == UNASSIGNED_LABEL

    threshold_prediction_tables[threshold] = prediction_df

    class_report = classification_report(
        y_test_text,
        final_labels,
        labels=PREDICTION_LABELS_WITH_UNASSIGNED,
        output_dict=True,
        zero_division=0,
    )
    threshold_class_reports[threshold] = pd.DataFrame(class_report).transpose()

    cm = confusion_matrix(
        y_test_text,
        final_labels,
        labels=PREDICTION_LABELS_WITH_UNASSIGNED,
    )

    cm_df = pd.DataFrame(
        cm,
        index=PREDICTION_LABELS_WITH_UNASSIGNED,
        columns=PREDICTION_LABELS_WITH_UNASSIGNED,
    )

    cm_norm_df = normalize_confusion_matrix_rows(cm_df)

    threshold_confusion_matrices[threshold] = cm_df
    threshold_confusion_matrices_norm[threshold] = cm_norm_df

    save_normalized_confusion_matrix_light_heatmap(
        cm_df,
        title=f"Pancreas Level 3 Balanced LR with Unassigned threshold {threshold:.2f}",
        output_path=FIGURES_DIR / f"pancreas_LR_balanced_level3_unassigned_threshold_{sheet_suffix}_heatmap.png",
    )

threshold_summary_df = pd.DataFrame(threshold_rows)

presentation_summary_df = create_presentation_summary(threshold_summary_df)

summary_table_path = FIGURES_DIR / "pancreas_LR_balanced_level3_unassigned_threshold_summary_table.png"
tradeoff_plot_path = FIGURES_DIR / "pancreas_LR_balanced_level3_unassigned_threshold_tradeoff_plot.png"
coverage_plot_path = FIGURES_DIR / "pancreas_LR_balanced_level3_unassigned_threshold_assigned_unassigned_plot.png"
save_threshold_summary_table(
    presentation_summary_df,
    summary_table_path,
    selected_threshold=SELECTED_THRESHOLD,
)
save_threshold_tradeoff_plot(threshold_summary_df, tradeoff_plot_path)
save_assigned_unassigned_bar_plot(threshold_summary_df, coverage_plot_path)

all_threshold_predictions = base_predictions.copy()
for threshold in THRESHOLDS:
    final_labels = apply_unassigned_threshold(
        predicted_labels,
        confidence_scores,
        threshold,
    )
    col_name = f"final_label_threshold_{threshold:.2f}".replace(".", "_")
    all_threshold_predictions[col_name] = final_labels

label_distribution_total = create_label_distribution_table(y_text, "total_dataset")
label_distribution_development = create_label_distribution_table(
    y_dev_text,
    "development_set_used_for_threshold_training",
)
label_distribution_test = create_label_distribution_table(
    y_test_text,
    "test_set_used_for_threshold_evaluation",
)

experiment_info = pd.DataFrame({
    "field": [
        "script_name",
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
        "thresholds_tested",
        "unassigned_label",
        "selected_threshold_for_table_highlight",
        "training_target_sum",
        "training_min_counts",
        "training_min_genes",
        "normalization",
        "log_transform",
        "feature_selection",
        "training_time_seconds",
        "prediction_time_seconds",
    ],
    "value": [
        "03_compare_unassigned_thresholds_pancreas.py",
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
        "None",
        ", ".join([str(t) for t in THRESHOLDS]),
        UNASSIGNED_LABEL,
        SELECTED_THRESHOLD,
        training_target_sum,
        training_min_counts,
        training_min_genes,
        "library_size_normalization",
        "log1p",
        "highly_variable_genes",
        training_time_seconds,
        prediction_time_seconds,
    ],
})

with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
    threshold_summary_df.to_excel(writer, sheet_name="threshold_summary", index=False)
    presentation_summary_df.to_excel(writer, sheet_name="presentation_summary", index=False)
    baseline_metrics_df.to_excel(writer, sheet_name="baseline_no_unassigned", index=False)
    all_threshold_predictions.to_excel(writer, sheet_name="test_predictions_all", index=False)

    label_distribution_total.to_excel(writer, sheet_name="label_distribution_total", index=False)
    label_distribution_development.to_excel(writer, sheet_name="label_distribution_development", index=False)
    label_distribution_test.to_excel(writer, sheet_name="label_distribution_test", index=False)
    experiment_info.to_excel(writer, sheet_name="experiment_info", index=False)

    for threshold in THRESHOLDS:
        sheet_suffix = f"{int(round(threshold * 100)):02d}"
        threshold_class_reports[threshold].to_excel(
            writer,
            sheet_name=f"report_t{sheet_suffix}",
        )
        threshold_confusion_matrices[threshold].to_excel(
            writer,
            sheet_name=f"cm_t{sheet_suffix}",
        )
        threshold_confusion_matrices_norm[threshold].to_excel(
            writer,
            sheet_name=f"cm_norm_t{sheet_suffix}",
        )

metadata = {
    "script_name": "03_compare_unassigned_thresholds_pancreas.py",
    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_path": str(DATA_PATH),
    "split_path": str(SPLIT_PATH),
    "label_column": LABEL_COLUMN,
    "model_type": "LogisticRegression",
    "class_weight": "balanced",
    "n_cells_total": int(N_CELLS),
    "n_features": int(N_FEATURES),
    "n_classes": int(N_CLASSES),
    "class_names": CLASS_NAMES,
    "unassigned_label": UNASSIGNED_LABEL,
    "thresholds_tested": THRESHOLDS,
    "selected_threshold_for_table_highlight": SELECTED_THRESHOLD,
    "development_set_size": int(N_DEV),
    "test_set_size": int(N_TEST),
    "training_time_seconds": training_time_seconds,
    "prediction_time_seconds": prediction_time_seconds,
    "baseline_test_metrics": baseline_metrics,
    "threshold_summary": threshold_summary_df.to_dict(orient="records"),
    "output_excel_path": str(EXCEL_PATH),
    "figures_dir": str(FIGURES_DIR),
    "presentation_figures": {
        "summary_table": str(summary_table_path),
        "tradeoff_plot": str(tradeoff_plot_path),
        "assigned_unassigned_plot": str(coverage_plot_path),
    },
}

save_json(metadata, METADATA_PATH)

print("\nThreshold comparison completed.")
print("\nMain threshold summary:")
print(threshold_summary_df[[
    "threshold",
    "assigned_percentage",
    "unassigned_percentage",
    "assigned_accuracy",
    "assigned_f1_macro",
    "strict_accuracy_with_unassigned_as_wrong",
]])

print("\nImportant saved files:")
print(f"Results:  {EXCEL_PATH}")
print(f"Metadata: {METADATA_PATH}")
print(f"Figures:  {FIGURES_DIR}")
