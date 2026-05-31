# create_gui_test_subset_from_raw.py

"""
Create a semi-balanced raw-count GUI/backend test subset from PBMC68k.

Purpose:
This script creates two .h5ad files:

1. pbmc68k_gui_test_2401cells.h5ad
   - unlabeled GUI test input
   - simulates a real user-uploaded dataset
   - cell type labels are removed from adata.obs

2. pbmc68k_gui_test_2401cells_labelled_for_control.h5ad
   - labelled control version
   - keeps true labels for debugging/checking only

Both files are created from pbmc68k_annotated_with_levels.h5ad,
where adata.X contains raw UMI counts.

These files are for MVP GUI/backend integration testing only.
They should NOT replace the full held-out test set for final model evaluation.
"""

from pathlib import Path

import numpy as np
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

# Main GUI input file: labels removed
OUTPUT_UNLABELED_PATH = OUTPUT_DIR / "pbmc68k_gui_test_2401cells.h5ad"

# Control/debug file: labels kept
OUTPUT_LABELLED_CONTROL_PATH = (
    OUTPUT_DIR / "pbmc68k_gui_test_2401cells_labelled_for_control.h5ad"
)

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

# These columns should NOT be present in the actual GUI input file.
# They are kept only in the labelled control file.
LABEL_COLUMNS_TO_REMOVE = [
    "cell_type",
    "cell_type_original",
    "cell_type_standardized",
    "cell_type_level_1",
    "cell_type_level_2",
    "cell_type_level_3",
]


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

adata_subset_labelled = adata_test[selected_obs_names].copy()


# ============================================================
# 5. Add metadata to labelled control file
# ============================================================

adata_subset_labelled.uns["gui_test_subset_info"] = {
    "purpose": (
        "Semi-balanced raw-count held-out test subset for GUI/backend MVP testing"
    ),
    "source_dataset": str(DATA_PATH),
    "split_source": str(SPLIT_PATH),
    "label_column_used_for_sampling": LABEL_COLUMN,
    "sampling_targets": SAMPLING_TARGETS,
    "random_state": RANDOM_STATE,
    "important_note": (
        "The labelled control file keeps true labels only for debugging/checking. "
        "The main GUI input file removes these labels to simulate a real "
        "user-uploaded dataset."
    ),
    "matrix_note": (
        "adata.X contains raw UMI counts from pbmc68k_annotated_with_levels.h5ad. "
        "This file is not normalized, log-transformed, or HVG-filtered."
    ),
}


# ============================================================
# 6. Create summary table before removing labels
# ============================================================

summary = (
    adata_subset_labelled.obs[LABEL_COLUMN]
    .value_counts()
    .rename_axis("class_name")
    .reset_index(name="count")
)

summary["percentage"] = summary["count"] / summary["count"].sum() * 100

print("\nFinal labelled control subset:")
print(adata_subset_labelled)

print("\nFinal Level 1 distribution:")
print(summary)

if "cell_type_level_2" in adata_subset_labelled.obs.columns:
    print("\nFinal Level 2 distribution:")
    print(adata_subset_labelled.obs["cell_type_level_2"].value_counts())

if "cell_type_original" in adata_subset_labelled.obs.columns:
    print("\nFinal original label distribution:")
    print(adata_subset_labelled.obs["cell_type_original"].value_counts())


# ============================================================
# 7. Create unlabeled GUI input file
# ============================================================

adata_subset_unlabeled = adata_subset_labelled.copy()

existing_label_columns = [
    col for col in LABEL_COLUMNS_TO_REMOVE
    if col in adata_subset_unlabeled.obs.columns
]

adata_subset_unlabeled.obs = adata_subset_unlabeled.obs.drop(
    columns=existing_label_columns
)

adata_subset_unlabeled.uns["gui_test_subset_info"][
    "label_removal_note"
] = (
    "Ground-truth cell type labels were removed from adata.obs in this "
    "main GUI input file to simulate a real user-uploaded dataset. "
    "A separate labelled control file is saved for debugging/checking."
)

adata_subset_unlabeled.uns["gui_test_subset_info"][
    "removed_label_columns"
] = existing_label_columns


# ============================================================
# 8. Save outputs
# ============================================================

# Save actual GUI input: unlabeled
adata_subset_unlabeled.write_h5ad(OUTPUT_UNLABELED_PATH)

# Save labelled control/debug version
adata_subset_labelled.write_h5ad(OUTPUT_LABELLED_CONTROL_PATH)

# Save summary table
summary.to_csv(SUMMARY_PATH, index=False)

print("\nSaved unlabeled GUI test subset to:")
print(OUTPUT_UNLABELED_PATH)

print("\nSaved labelled control file to:")
print(OUTPUT_LABELLED_CONTROL_PATH)

print("\nSaved summary table to:")
print(SUMMARY_PATH)

print("\nRemaining obs columns in unlabeled GUI input:")
print(list(adata_subset_unlabeled.obs.columns))

print("\nObs columns in labelled control file:")
print(list(adata_subset_labelled.obs.columns))
