# create_gui_test_subset_from_raw.py

"""
Create a semi-balanced raw-count GUI/backend test subset from PBMC68k.

Purpose:
This script creates a small .h5ad file that behaves like a user-uploaded dataset.
It uses pbmc68k_annotated_with_levels.h5ad, where adata.X contains the raw UMI count matrix.

This file is for MVP GUI/backend integration testing only.
It should NOT be used as the final model performance evaluation dataset.

Output:
model_development/GUI_test_subset/pbmc68k_gui_test_3000cells.h5ad
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


# ============================================================
# 1. Paths and settings
# ============================================================

RANDOM_STATE = 42
LABEL_COLUMN = "cell_type_level_1"

# Raw/annotated dataset, not preprocessed.
DATA_PATH = Path(
    r"C:\Users\ferid\Downloads\capstone_demo\pbmc68k_annotated_with_levels.h5ad"
)

PROJECT_DIR = Path(r"C:\Users\ferid\Downloads\capstone_demo\model_development")

# Saved split indices from the ML experiments
SPLIT_PATH = (
    PROJECT_DIR
    / "results"
    / "LR_level1"
    / "LR_level1_split_indices.npz"
)

OUTPUT_DIR = PROJECT_DIR / "GUI_test_subset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "pbmc68k_gui_test_2401cells.h5ad"
SUMMARY_PATH = OUTPUT_DIR / "pbmc68k_gui_test_2401cells_summary.csv"


# Semi-balanced sampling plan from the held-out test set.
# If fewer cells are available, the script takes all available cells.
SAMPLING_TARGETS = {
    "B cell": 500,
    "Dendritic": 355,
    "Monocyte": 500,
    "NK cell": 500,
    "Progenitor": 46,
    "T cell": 500,
}


# ============================================================
# 2. Load raw/annotated dataset
# ============================================================

print("Loading raw annotated dataset...")
adata = sc.read_h5ad(DATA_PATH)

print("\nLoaded dataset:")
print(adata)

if LABEL_COLUMN not in adata.obs.columns:
    raise ValueError(f"Column '{LABEL_COLUMN}' was not found in adata.obs.")

if not SPLIT_PATH.exists():
    raise FileNotFoundError(
        f"Split file not found:\n{SPLIT_PATH}\n"
        "Please make sure LR_level1_split_indices.npz exists."
    )

# Make sure X is raw count-like, not already HVG-filtered.
print("\nMatrix shape:")
print(adata.shape)

if adata.n_vars < 10000:
    print(
        "\nWARNING: This dataset has fewer than 10,000 genes. "
        "Please check that you are using the raw annotated dataset, "
        "not the preprocessed HVG-filtered training dataset."
    )

if sparse.issparse(adata.X):
    print("adata.X is sparse.")
else:
    print("adata.X is dense.")

print("\nLevel 1 distribution in full raw dataset:")
print(adata.obs[LABEL_COLUMN].value_counts())


# ============================================================
# 3. Load held-out test indices
# ============================================================

split_data = np.load(SPLIT_PATH)
test_indices = split_data["test_indices"]

adata_test = adata[test_indices].copy()

print("\nHeld-out test subset from raw dataset:")
print(adata_test)

print("\nLevel 1 distribution in held-out test set:")
print(adata_test.obs[LABEL_COLUMN].value_counts())


# ============================================================
# 4. Semi-balanced sampling
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

selected_obs_names = []

print("\nSampling cells:")

for class_name, target_n in SAMPLING_TARGETS.items():
    class_cells = adata_test.obs_names[
        adata_test.obs[LABEL_COLUMN].astype(str) == class_name
    ].to_numpy()

    n_available = len(class_cells)
    n_to_sample = min(target_n, n_available)

    if n_available == 0:
        print(f"{class_name}: no cells available, skipping.")
        continue

    sampled_cells = rng.choice(
        class_cells,
        size=n_to_sample,
        replace=False,
    )

    selected_obs_names.extend(sampled_cells)

    print(f"{class_name}: selected {n_to_sample} / {n_available}")


# Shuffle cells so they are not grouped by class in the output file
selected_obs_names = np.array(selected_obs_names)
rng.shuffle(selected_obs_names)

adata_subset = adata_test[selected_obs_names].copy()


# ============================================================
# 5. Add metadata
# ============================================================

adata_subset.uns["gui_test_subset_info"] = {
    "purpose": "Semi-balanced raw-count held-out test subset for GUI/backend MVP testing",
    "source_dataset": str(DATA_PATH),
    "split_source": str(SPLIT_PATH),
    "label_column_used_for_sampling": LABEL_COLUMN,
    "sampling_targets": SAMPLING_TARGETS,
    "random_state": RANDOM_STATE,
    "important_note": (
        "This subset is intended for software integration testing only. "
        "It is created from the raw annotated dataset so that the full inference "
        "workflow can be tested: data loading, validation, inference preprocessing, "
        "gene alignment, model prediction, and GUI output. It should not replace "
        "the full held-out test set for final model performance evaluation."
    ),
    "matrix_note": (
        "adata.X contains raw UMI counts from pbmc68k_annotated_with_levels.h5ad. "
        "This file is not normalized, log-transformed, or HVG-filtered."
    ),
}


# ============================================================
# 6. Create summary table
# ============================================================

summary = (
    adata_subset.obs[LABEL_COLUMN]
    .value_counts()
    .rename_axis("class_name")
    .reset_index(name="count")
)

summary["percentage"] = summary["count"] / summary["count"].sum() * 100

print("\nFinal GUI test subset:")
print(adata_subset)

print("\nFinal Level 1 distribution:")
print(summary)

if "cell_type_level_2" in adata_subset.obs.columns:
    print("\nFinal Level 2 distribution:")
    print(adata_subset.obs["cell_type_level_2"].value_counts())

if "cell_type_original" in adata_subset.obs.columns:
    print("\nFinal original label distribution:")
    print(adata_subset.obs["cell_type_original"].value_counts())


# ============================================================
# 7. Save output
# ============================================================

adata_subset.write_h5ad(OUTPUT_PATH)
summary.to_csv(SUMMARY_PATH, index=False)

print("\nSaved GUI test subset to:")
print(OUTPUT_PATH)

print("\nSaved summary table to:")
print(SUMMARY_PATH)