# 04_compare_all_models.py

"""
Combine model comparison results for Level 1 cell-type prediction.

This script does NOT retrain models.
It reads the existing CV result Excel files for LR, RF, and SVM,
combines their summary_metrics sheets, and creates slide-friendly figures.

Inputs:
- LR_level1_CV_results.xlsx
- RF_level1_CV_results.xlsx
- SVM_level1_CV_results.xlsx

Outputs:
- all_models_level1_comparison.xlsx
- all_models_level1_summary_table.png
- all_models_level1_macro_f1_barplot.png
- all_models_level1_accuracy_barplot.png
- all_models_level1_balanced_accuracy_barplot.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Paths
# ============================================================

PROJECT_DIR = Path(r"C:\Users\ferid\Downloads\capstone_demo\model_development")

RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "all_models_level1"
FIGURES_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LR_PATH = RESULTS_DIR / "LR_level1" / "LR_level1_CV_results.xlsx"
RF_PATH = RESULTS_DIR / "RF_level1" / "RF_level1_CV_results.xlsx"
SVM_PATH = RESULTS_DIR / "SVM_level1" / "SVM_level1_CV_results.xlsx"

OUTPUT_EXCEL = OUTPUT_DIR / "all_models_level1_comparison.xlsx"


# ============================================================
# 2. Helper functions
# ============================================================

def read_summary(path, model_family):
    """
    Read summary_metrics sheet from one model result workbook.
    """
    if not path.exists():
        raise FileNotFoundError(f"Could not find file:\n{path}")

    df = pd.read_excel(path, sheet_name="summary_metrics")
    df.insert(0, "model_family", model_family)
    return df


def save_barplot(df, metric_col, title, ylabel, output_path):
    """
    Save a slide-friendly barplot for one metric.
    """
    plot_df = df.sort_values(metric_col, ascending=False).copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(plot_df["model_name"], plot_df[metric_col])

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1)

    for i, value in enumerate(plot_df[metric_col]):
        ax.text(
            i,
            value + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_summary_table_image(df, output_path):
    """
    Save a compact model comparison table as PNG for PowerPoint.
    """
    important_cols = [
        "model_name",
        "algorithm",
        "class_weight",
        "accuracy_mean",
        "balanced_accuracy_mean",
        "f1_macro_mean",
        "f1_weighted_mean",
        "recall_macro_mean",
        "training_time_seconds_mean",
    ]

    existing_cols = [col for col in important_cols if col in df.columns]
    table_df = df[existing_cols].copy()

    table_df = table_df.sort_values("f1_macro_mean", ascending=False)

    rename_dict = {
        "model_name": "Model",
        "algorithm": "Algorithm",
        "class_weight": "Class Weight",
        "accuracy_mean": "Accuracy",
        "balanced_accuracy_mean": "Balanced Acc.",
        "f1_macro_mean": "Macro F1",
        "f1_weighted_mean": "Weighted F1",
        "recall_macro_mean": "Macro Recall",
        "training_time_seconds_mean": "Train Time (s)",
    }

    table_df = table_df.rename(columns=rename_dict)

    for col in table_df.columns:
        if col not in ["Model", "Algorithm", "Class Weight"]:
            table_df[col] = table_df[col].apply(
                lambda x: f"{x:.3f}" if isinstance(x, (int, float, np.floating)) else x
            )

    fig, ax = plt.subplots(figsize=(15, 4.2))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)

    # Make header bold
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_text_props(weight="bold")

    ax.set_title(
        "Level 1 Model Comparison Based on 5-Fold Cross-Validation",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# 3. Read and combine model summaries
# ============================================================

lr_summary = read_summary(LR_PATH, "LR")
rf_summary = read_summary(RF_PATH, "RF")
svm_summary = read_summary(SVM_PATH, "SVM")

combined = pd.concat(
    [lr_summary, rf_summary, svm_summary],
    ignore_index=True,
)

combined = combined.sort_values("f1_macro_mean", ascending=False)


# ============================================================
# 4. Create compact presentation table
# ============================================================

presentation_columns = [
    "model_name",
    "algorithm",
    "class_weight",
    "accuracy_mean",
    "balanced_accuracy_mean",
    "f1_macro_mean",
    "f1_weighted_mean",
    "recall_macro_mean",
    "training_time_seconds_mean",
]

presentation_table = combined[presentation_columns].copy()
presentation_table = presentation_table.sort_values("f1_macro_mean", ascending=False)


# ============================================================
# 5. Save Excel workbook
# ============================================================

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    combined.to_excel(writer, sheet_name="all_summary_metrics", index=False)
    presentation_table.to_excel(writer, sheet_name="presentation_table", index=False)

    lr_summary.to_excel(writer, sheet_name="LR_summary", index=False)
    rf_summary.to_excel(writer, sheet_name="RF_summary", index=False)
    svm_summary.to_excel(writer, sheet_name="SVM_summary", index=False)


# ============================================================
# 6. Save figures
# ============================================================

save_summary_table_image(
    presentation_table,
    FIGURES_DIR / "all_models_level1_summary_table.png",
)

save_barplot(
    combined,
    metric_col="f1_macro_mean",
    title="Level 1 Model Comparison: Macro F1",
    ylabel="Mean Macro F1",
    output_path=FIGURES_DIR / "all_models_level1_macro_f1_barplot.png",
)

save_barplot(
    combined,
    metric_col="accuracy_mean",
    title="Level 1 Model Comparison: Accuracy",
    ylabel="Mean Accuracy",
    output_path=FIGURES_DIR / "all_models_level1_accuracy_barplot.png",
)

save_barplot(
    combined,
    metric_col="balanced_accuracy_mean",
    title="Level 1 Model Comparison: Balanced Accuracy",
    ylabel="Mean Balanced Accuracy",
    output_path=FIGURES_DIR / "all_models_level1_balanced_accuracy_barplot.png",
)


# ============================================================
# 7. Print result
# ============================================================

print("\nCombined model comparison finished.")
print("\nModels ranked by Macro F1:")
print(
    presentation_table[
        [
            "model_name",
            "accuracy_mean",
            "balanced_accuracy_mean",
            "f1_macro_mean",
            "f1_weighted_mean",
            "recall_macro_mean",
            "training_time_seconds_mean",
        ]
    ]
)

best_model = presentation_table.iloc[0]["model_name"]
best_macro_f1 = presentation_table.iloc[0]["f1_macro_mean"]

print(f"\nBest model by Macro F1: {best_model}")
print(f"Best Macro F1: {best_macro_f1:.4f}")

print("\nSaved files:")
print(f"Excel workbook: {OUTPUT_EXCEL}")
print(f"Figures folder: {FIGURES_DIR}")