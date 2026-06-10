from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import shap
from scipy import sparse


# ======================================================================
# Project paths
# ======================================================================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

INFERENCE_PREPROCESSING_DIR = PROJECT_ROOT / "src" / "Inference_Preprocessing"
if str(INFERENCE_PREPROCESSING_DIR) not in sys.path:
    sys.path.append(str(INFERENCE_PREPROCESSING_DIR))

from preprocess_inference import preprocess_for_inference
from gene_alignment import align_genes_to_training


MODEL_BUNDLE_PATH = (
    PROJECT_ROOT
    / "src"
    / "ml_model"
    / "models"
    / "LR_level1_no_weight_final_model_bundle.joblib"
)

INPUT_H5AD_PATH = (
    PROJECT_ROOT
    / "data"
    / "pbmc68k_gui_test_2401cells.h5ad"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shap_LR_level1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# SHAP settings
# ======================================================================

RANDOM_STATE = 42

# Keep these small enough for fast testing.
# You can increase EXPLAIN_SIZE later if needed.
BACKGROUND_SIZE = 100
SHAP_BATCH_SIZE = 256
SAVE_CELL_LEVEL_OUTPUT = False

# Global/class-level plots
TOP_N_GENES = 20

# GUI/report-oriented outputs
TOP_N_GROUP_PLOT_GENES = 10
TOP_N_CELL_GENES = 10


# ======================================================================
# Helper functions
# ======================================================================

def to_dense_array(X):
    """
    Convert sparse matrices to dense NumPy arrays.

    The final test dataset has 2401 cells x 2000 genes after alignment,
    so dense conversion is acceptable for this SHAP analysis.
    """
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def sample_rows(X: np.ndarray, max_rows: int, random_state: int):
    """
    Sample up to max_rows cells from X.
    """
    n_rows = X.shape[0]

    if n_rows <= max_rows:
        idx = np.arange(n_rows)
        return X, idx

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n_rows, size=max_rows, replace=False)
    idx = np.sort(idx)

    return X[idx], idx


def normalize_shap_output(shap_values):
    """
    Convert SHAP output to a consistent shape:
        cells x genes x classes

    Depending on the SHAP version, multiclass LinearExplainer may return:
    - a list of arrays, one per class
    - a 3D NumPy array
    """
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=-1)

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 2:
        return shap_values[:, :, np.newaxis]

    if shap_values.ndim == 3:
        return shap_values

    raise ValueError(f"Unexpected SHAP output shape: {shap_values.shape}")

def compute_shap_values_in_batches(explainer, X, batch_size: int = 256):
    """
    Compute SHAP values for all cells in batches to reduce peak memory usage.
    Returns an array with shape: cells x genes x classes
    """
    chunks = []

    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        batch_X = X[start:end]

        batch_shap = explainer.shap_values(batch_X)
        batch_shap = normalize_shap_output(batch_shap)

        chunks.append(batch_shap)

    return np.concatenate(chunks, axis=0)


def normalize_gene_symbols(gene_symbols, training_gene_order):
    """
    Convert gene_symbols from the model package into a list aligned with
    training_gene_order.

    Supports:
    - None
    - dict mapping Ensembl ID -> gene symbol
    - list / tuple / pandas Series / numpy array
    """
    if gene_symbols is None:
        return [None] * len(training_gene_order)

    if isinstance(gene_symbols, dict):
        return [gene_symbols.get(gene_id) for gene_id in training_gene_order]

    gene_symbols = list(gene_symbols)

    if len(gene_symbols) != len(training_gene_order):
        raise ValueError(
            f"gene_symbols length mismatch: got {len(gene_symbols)}, "
            f"expected {len(training_gene_order)}."
        )

    return gene_symbols


def get_class_index_map(class_names):
    """
    Build a mapping from class name to class index in the SHAP output.
    """
    return {class_name: idx for idx, class_name in enumerate(class_names)}


def safe_filename(name: str) -> str:
    """
    Make class names safe to use in filenames.
    """
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
    )


def make_gene_label(gene_symbol, ensembl_id: str) -> str:
    """
    Prefer gene symbols in plots, but keep Ensembl IDs available in tables.
    """
    if gene_symbol is None or pd.isna(gene_symbol) or str(gene_symbol).strip() == "":
        return str(ensembl_id)
    return str(gene_symbol)


# ======================================================================
# 1. Load final model package
# ======================================================================

print("Loading final Logistic Regression model package...")

model_package = joblib.load(MODEL_BUNDLE_PATH)

model = model_package["model"]
label_encoder = model_package["label_encoder"]
class_names = list(model_package["class_names"])
training_gene_order = list(model_package["training_gene_order"])
gene_symbols = normalize_gene_symbols(
    model_package.get("gene_symbols"),
    training_gene_order,
)

preprocessing = model_package["preprocessing"]
target_sum = preprocessing["target_sum"]
min_counts = preprocessing["min_counts"]
min_genes = preprocessing["min_genes"]

print(f"Model name: {model_package.get('model_name')}")
print(f"Model type: {model_package.get('model_type')}")
print(f"Classes: {class_names}")
print(f"Number of training genes: {len(training_gene_order)}")
print(f"Using training target_sum: {target_sum}")
print(f"Using min_counts: {min_counts}")
print(f"Using min_genes: {min_genes}")


# ======================================================================
# 2. Load test/query dataset
# ======================================================================

print("Loading query dataset...")

adata = sc.read_h5ad(INPUT_H5AD_PATH)

print(f"Input AnnData: {adata.n_obs} cells x {adata.n_vars} genes")


# ======================================================================
# 3. Preprocess with training preprocessing parameters
# ======================================================================

print("Preprocessing query data using saved training parameters...")

adata_pp = preprocess_for_inference(
    adata=adata,
    min_counts=min_counts,
    min_genes=min_genes,
    target_sum=target_sum,
)


# ======================================================================
# 4. Align genes to exact training HVG order
# ======================================================================

print("Aligning query data to saved training gene order...")

adata_aligned = align_genes_to_training(
    adata=adata_pp,
    training_gene_order=training_gene_order,
)

if adata_aligned.n_vars != len(training_gene_order):
    raise ValueError(
        f"Feature count mismatch: aligned data has {adata_aligned.n_vars} genes, "
        f"but the model expects {len(training_gene_order)} genes."
    )

if list(adata_aligned.var_names.astype(str)) != training_gene_order:
    raise ValueError(
        "Gene order mismatch after alignment. "
        "SHAP must use the exact training_gene_order from the model package."
    )

print(f"Aligned AnnData: {adata_aligned.n_obs} cells x {adata_aligned.n_vars} genes")


# ======================================================================
# 5. Predict cell types
# ======================================================================

print("Running model predictions...")

X = to_dense_array(adata_aligned.X)

pred_encoded = model.predict(X)
pred_labels = label_encoder.inverse_transform(pred_encoded)

prediction_df = pd.DataFrame(
    {
        "cell_id": adata_aligned.obs_names.astype(str),
        "predicted_cell_type": pred_labels,
    }
)

prediction_path = OUTPUT_DIR / "predicted_cell_types.csv"
prediction_df.to_csv(prediction_path, index=False)

print(f"Saved predictions to: {prediction_path}")


# ======================================================================
# 6. Prepare SHAP background data
# ======================================================================

print("Preparing background data for SHAP...")

background, _ = sample_rows(
    X,
    max_rows=BACKGROUND_SIZE,
    random_state=RANDOM_STATE,
)

X_explain = X
explained_cell_ids = adata_aligned.obs_names.astype(str).to_numpy()
explained_predictions = pred_labels

print(f"Background data shape: {background.shape}")
print(f"Explained data shape: {X_explain.shape}")


# ======================================================================
# 7. Run SHAP LinearExplainer
# ======================================================================

print("Running SHAP LinearExplainer...")

explainer = shap.LinearExplainer(model, background)

shap_values = compute_shap_values_in_batches(
    explainer,
    X_explain,
    batch_size=SHAP_BATCH_SIZE,
)

print(f"SHAP values shape: {shap_values.shape}")
print("SHAP shape meaning: cells x genes x classes")

class_to_index = get_class_index_map(class_names)


# ======================================================================
# 8. Save SHAP values for explained cells metadata
# ======================================================================

explained_cells_df = pd.DataFrame(
    {
        "cell_id": explained_cell_ids,
        "predicted_cell_type": explained_predictions,
    }
)

explained_cells_path = OUTPUT_DIR / "shap_explained_cells.csv"
explained_cells_df.to_csv(explained_cells_path, index=False)

print(f"Saved explained cell metadata to: {explained_cells_path}")


# ======================================================================
# 9. Global SHAP gene importance
# ======================================================================

print("Computing global SHAP gene importance...")

global_importance = np.mean(np.abs(shap_values), axis=(0, 2))

global_df = pd.DataFrame(
    {
        "feature_order": np.arange(len(training_gene_order)),
        "ensembl_id": training_gene_order,
        "gene_symbol": gene_symbols,
        "mean_abs_shap": global_importance,
    }
)

global_df = global_df.sort_values("mean_abs_shap", ascending=False)

global_csv_path = OUTPUT_DIR / "shap_global_gene_importance.csv"
global_df.to_csv(global_csv_path, index=False)

print(f"Saved global SHAP importance to: {global_csv_path}")


# ======================================================================
# 10. Class-specific SHAP gene importance
# ======================================================================

print("Computing class-specific SHAP gene importance...")

class_tables = []

for class_idx, class_name in enumerate(class_names):
    class_importance = np.mean(np.abs(shap_values[:, :, class_idx]), axis=0)

    class_df = pd.DataFrame(
        {
            "class_name": class_name,
            "feature_order": np.arange(len(training_gene_order)),
            "ensembl_id": training_gene_order,
            "gene_symbol": gene_symbols,
            "mean_abs_shap": class_importance,
        }
    )

    class_df = class_df.sort_values("mean_abs_shap", ascending=False)
    class_tables.append(class_df)

    class_csv_path = OUTPUT_DIR / f"shap_top_genes_{safe_filename(class_name)}.csv"
    class_df.to_csv(class_csv_path, index=False)

class_specific_df = pd.concat(class_tables, axis=0)

class_specific_csv_path = OUTPUT_DIR / "shap_class_specific_gene_importance.csv"
class_specific_df.to_csv(class_specific_csv_path, index=False)

print(f"Saved class-specific SHAP importance to: {class_specific_csv_path}")


# ======================================================================
# 11. Predicted-group SHAP gene importance
# ======================================================================

print("Computing predicted-group SHAP gene importance...")

predicted_group_tables = []

for predicted_group in sorted(pd.unique(explained_predictions)):
    if predicted_group not in class_to_index:
        raise ValueError(
            f"Predicted group '{predicted_group}' was not found in class_names."
        )

    group_mask = explained_predictions == predicted_group
    n_cells_group = int(np.sum(group_mask))
    class_idx = class_to_index[predicted_group]

    # For cells predicted as this group, summarize SHAP values for that same class.
    group_shap = shap_values[group_mask, :, class_idx]

    mean_abs_shap = np.mean(np.abs(group_shap), axis=0)
    mean_shap = np.mean(group_shap, axis=0)

    group_df = pd.DataFrame(
        {
            "predicted_group": predicted_group,
            "ensembl_id": training_gene_order,
            "gene_symbol": gene_symbols,
            "mean_abs_shap": mean_abs_shap,
            "mean_shap": mean_shap,
            "n_cells": n_cells_group,
        }
    )

    group_df = group_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    group_df.insert(1, "rank", np.arange(1, len(group_df) + 1))

    predicted_group_tables.append(group_df)

predicted_group_df = pd.concat(predicted_group_tables, axis=0)

predicted_group_csv_path = OUTPUT_DIR / "shap_gene_importance_by_predicted_group.csv"
predicted_group_df.to_csv(predicted_group_csv_path, index=False)

print(f"Saved predicted-group SHAP importance to: {predicted_group_csv_path}")


# ======================================================================
# 12. Cell-level SHAP explanations
# ======================================================================

if SAVE_CELL_LEVEL_OUTPUT:
    print("Computing cell-level SHAP explanations...")

    cell_level_rows = []

    for local_cell_idx, cell_id in enumerate(explained_cell_ids):
        predicted_class = explained_predictions[local_cell_idx]

        if predicted_class not in class_to_index:
            raise ValueError(
                f"Predicted class '{predicted_class}' was not found in class_names."
            )

        class_idx = class_to_index[predicted_class]

        cell_shap = shap_values[local_cell_idx, :, class_idx]
        abs_cell_shap = np.abs(cell_shap)

        top_gene_indices = np.argsort(abs_cell_shap)[::-1][:TOP_N_CELL_GENES]

        for rank, gene_idx in enumerate(top_gene_indices, start=1):
            cell_level_rows.append(
                {
                    "cell_id": cell_id,
                    "predicted_class": predicted_class,
                    "rank": rank,
                    "ensembl_id": training_gene_order[gene_idx],
                    "gene_symbol": gene_symbols[gene_idx],
                    "shap_value": float(cell_shap[gene_idx]),
                    "abs_shap_value": float(abs_cell_shap[gene_idx]),
                    "model_input_expression": float(X_explain[local_cell_idx, gene_idx]),
                }
            )

    cell_level_df = pd.DataFrame(cell_level_rows)
    cell_level_csv_path = OUTPUT_DIR / "shap_top_genes_per_cell.csv"
    cell_level_df.to_csv(cell_level_csv_path, index=False)

    print(f"Saved cell-level SHAP explanations to: {cell_level_csv_path}")


# ======================================================================
# 13. Plot global top SHAP genes
# ======================================================================

print("Saving SHAP plots...")

top_global = global_df.head(TOP_N_GENES).iloc[::-1].copy()
top_global["plot_label"] = [
    make_gene_label(symbol, ens_id)
    for symbol, ens_id in zip(top_global["gene_symbol"], top_global["ensembl_id"])
]

plt.figure(figsize=(8, 6))
plt.barh(top_global["plot_label"], top_global["mean_abs_shap"])
plt.xlabel("Mean absolute SHAP value")
plt.ylabel("Gene")
plt.title("Top global SHAP genes - Logistic Regression")
plt.tight_layout()

global_plot_path = OUTPUT_DIR / "shap_global_top_genes.png"
plt.savefig(global_plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved global SHAP plot to: {global_plot_path}")


# ======================================================================
# 14. Plot class-specific top SHAP genes
# ======================================================================

for class_name in class_names:
    class_df = class_specific_df[class_specific_df["class_name"] == class_name]
    top_class = class_df.head(TOP_N_GENES).iloc[::-1].copy()

    top_class["plot_label"] = [
        make_gene_label(symbol, ens_id)
        for symbol, ens_id in zip(top_class["gene_symbol"], top_class["ensembl_id"])
    ]

    plt.figure(figsize=(8, 6))
    plt.barh(top_class["plot_label"], top_class["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Gene")
    plt.title(f"Top SHAP genes for {class_name}")
    plt.tight_layout()

    class_plot_path = OUTPUT_DIR / f"shap_top_genes_{safe_filename(class_name)}.png"
    plt.savefig(class_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved class-specific SHAP plot to: {class_plot_path}")


# ======================================================================
# 15. Plot top SHAP genes by predicted group
# ======================================================================

print("Saving predicted-group SHAP plots...")

for predicted_group in sorted(pd.unique(explained_predictions)):
    group_df = predicted_group_df[
        predicted_group_df["predicted_group"] == predicted_group
    ]

    top_group = group_df.head(TOP_N_GROUP_PLOT_GENES).iloc[::-1].copy()

    top_group["plot_label"] = [
        make_gene_label(symbol, ens_id)
        for symbol, ens_id in zip(top_group["gene_symbol"], top_group["ensembl_id"])
    ]

    plt.figure(figsize=(8, 5))
    plt.barh(top_group["plot_label"], top_group["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Gene")
    plt.title(f"Top genes contributing to {predicted_group} predictions")
    plt.tight_layout()

    group_plot_path = (
        OUTPUT_DIR
        / f"shap_top_genes_by_predicted_group_{safe_filename(predicted_group)}.png"
    )
    plt.savefig(group_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved predicted-group SHAP plot to: {group_plot_path}")


print("SHAP analysis finished successfully.")
print(f"All SHAP outputs were saved to: {OUTPUT_DIR}")
