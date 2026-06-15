# create_gui_test_subset_pancreas.py

"""
Create a raw/full-gene GUI/backend test set from the pancreas held-out test split.

This script creates two .h5ad files:

1. pancreas_gui_test_all_test_cells.h5ad
   - unlabeled GUI test input
   - true cell type labels are removed from adata.obs

2. pancreas_gui_test_all_test_cells_labelled_for_control.h5ad
   - labelled control version
   - keeps true labels for debugging/checking only

The test cells are selected using the same 20% test indices from the pancreas
model comparison experiment. The GUI input itself is created from the raw/full-gene
annotated pancreas dataset, not from the preprocessed 2000-HVG training file.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


# Paths and settings
RANDOM_STATE = 42
LABEL_COLUMN = "cell_type_level_3"

PANCREAS_DIR = Path(
    r"C:\Users\ferid\Downloads\capstone_demo\pancreas_model"
)

RAW_ANNOTATED_PATH = (
    PANCREAS_DIR
    / "Tabula_Sapiens_Pancreas_annotated_with_levels.h5ad"
)

PREPROCESSED_PATH = (
    PANCREAS_DIR
    / "03_preprocess_train_pancreas_outputs"
    / "Tabula_Sapiens_Pancreas_preprocessed_for_training.h5ad"
)

SPLIT_PATH = (
    PANCREAS_DIR
    / "pancreas_model_development"
    / "results"
    / "all_models_level3"
    / "pancreas_all_models_level3_split_indices.npz"
)

OUTPUT_DIR = PANCREAS_DIR / "GUI_test_subset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_UNLABELED_PATH = OUTPUT_DIR / "pancreas_gui_test_all_test_cells.h5ad"
OUTPUT_LABELLED_CONTROL_PATH = (
    OUTPUT_DIR / "pancreas_gui_test_all_test_cells_labelled_for_control.h5ad"
)
SUMMARY_PATH = OUTPUT_DIR / "pancreas_gui_test_all_test_cells_summary.csv"

# Use all held-out test cells by default.
USE_ALL_TEST_SET = True

# Used only if USE_ALL_TEST_SET is False.
N_CELLS_TO_SAMPLE = 1000

LABEL_COLUMNS_TO_REMOVE = [
    "cell_type",
    "cell_type_original",
    "cell_type_standardized",
    "cell_type_level_1",
    "cell_type_level_2",
    "cell_type_level_3",
    "compartment",
    "broad_cell_class",
    "cell_ontology_class",
    "cell_type_ontology_term_id",
]


# Helpers
def stratified_sample_obs_names(adata, label_column, n_cells, random_state):
    labels = adata.obs[label_column].astype(str)
    rng = np.random.default_rng(random_state)

    class_counts = labels.value_counts()
    target_counts = (class_counts / class_counts.sum() * n_cells).round().astype(int)
    target_counts[target_counts == 0] = 1

    selected = []

    for class_name, target_n in target_counts.items():
        class_obs_names = adata.obs_names[labels == class_name].to_numpy()
        n_to_sample = min(int(target_n), len(class_obs_names))

        sampled = rng.choice(
            class_obs_names,
            size=n_to_sample,
            replace=False,
        )
        selected.extend(sampled)

    selected = np.array(selected)

    if len(selected) > n_cells:
        selected = rng.choice(selected, size=n_cells, replace=False)

    rng.shuffle(selected)
    return selected


def create_summary(adata, label_column):
    summary = (
        adata.obs[label_column]
        .astype(str)
        .value_counts()
        .rename_axis("class_name")
        .reset_index(name="count")
    )
    summary["percentage"] = summary["count"] / summary["count"].sum() * 100
    return summary


def check_path(path, description):
    if not path.exists():
        raise FileNotFoundError(f"{description} was not found:\n{path}")


# Check files
check_path(RAW_ANNOTATED_PATH, "Raw annotated pancreas dataset")
check_path(PREPROCESSED_PATH, "Preprocessed pancreas dataset")
check_path(SPLIT_PATH, "Saved pancreas split file")


# Load datasets
print("Loading raw/full-gene annotated pancreas dataset...")
adata_raw = sc.read_h5ad(RAW_ANNOTATED_PATH)
print(adata_raw)

print("\nLoading preprocessed pancreas dataset for test-cell mapping...")
adata_pp = sc.read_h5ad(PREPROCESSED_PATH)
print(adata_pp)

if LABEL_COLUMN not in adata_raw.obs.columns:
    raise ValueError(f"Column '{LABEL_COLUMN}' was not found in raw adata.obs.")

if LABEL_COLUMN not in adata_pp.obs.columns:
    raise ValueError(f"Column '{LABEL_COLUMN}' was not found in preprocessed adata.obs.")

if adata_raw.obs_names.has_duplicates:
    raise ValueError("Raw annotated dataset has duplicate obs_names.")

if adata_pp.obs_names.has_duplicates:
    raise ValueError("Preprocessed dataset has duplicate obs_names.")

if sparse.issparse(adata_raw.X):
    adata_raw.X = adata_raw.X.tocsr()

print("\nRaw dataset Level 3 distribution:")
print(adata_raw.obs[LABEL_COLUMN].value_counts())

print("\nPreprocessed dataset Level 3 distribution:")
print(adata_pp.obs[LABEL_COLUMN].value_counts())


# Load held-out test split
split_data = np.load(SPLIT_PATH)
test_indices = split_data["test_indices"]

if test_indices.max() >= adata_pp.n_obs:
    raise ValueError(
        "The saved test indices are outside the size of the preprocessed dataset."
    )

# Convert row indices from the preprocessed dataset to cell IDs.
test_obs_names = adata_pp.obs_names[test_indices].to_numpy()

missing_cells = [cell for cell in test_obs_names if cell not in adata_raw.obs_names]
if len(missing_cells) > 0:
    raise ValueError(
        f"{len(missing_cells)} held-out test cells from the preprocessed dataset "
        "were not found in the raw annotated dataset."
    )

adata_test = adata_raw[test_obs_names].copy()

print("\nHeld-out test set selected from raw/full-gene dataset:")
print(adata_test)

print("\nHeld-out test Level 3 distribution:")
print(adata_test.obs[LABEL_COLUMN].value_counts())


# Choose all test cells or a smaller stratified sample
if USE_ALL_TEST_SET:
    adata_subset_labelled = adata_test.copy()
    sampling_note = "All held-out test cells were used."
else:
    selected_obs_names = stratified_sample_obs_names(
        adata=adata_test,
        label_column=LABEL_COLUMN,
        n_cells=N_CELLS_TO_SAMPLE,
        random_state=RANDOM_STATE,
    )
    adata_subset_labelled = adata_test[selected_obs_names].copy()
    sampling_note = f"A stratified sample of about {N_CELLS_TO_SAMPLE} cells was used."

print("\nFinal labelled control subset:")
print(adata_subset_labelled)


# Add metadata
adata_subset_labelled.uns["gui_test_subset_info"] = {
    "purpose": "Raw/full-gene held-out test set for pancreas GUI/backend testing",
    "source_dataset": str(RAW_ANNOTATED_PATH),
    "preprocessed_dataset_used_only_for_mapping": str(PREPROCESSED_PATH),
    "split_source": str(SPLIT_PATH),
    "label_column_used_for_summary": LABEL_COLUMN,
    "use_all_test_set": USE_ALL_TEST_SET,
    "n_cells_to_sample_if_not_all": N_CELLS_TO_SAMPLE,
    "sampling_note": sampling_note,
    "random_state": RANDOM_STATE,
    "important_note": (
        "The labelled control file keeps true labels only for debugging/checking. "
        "The main GUI input file removes these labels to simulate a real upload."
    ),
    "matrix_note": (
        "adata.X comes from Tabula_Sapiens_Pancreas_annotated_with_levels.h5ad. "
        "This is the raw/count-like full-gene matrix, not normalized, not log-transformed, "
        "and not HVG-filtered."
    ),
    "n_cells": int(adata_subset_labelled.n_obs),
    "n_genes": int(adata_subset_labelled.n_vars),
}


# Save summary before removing labels
summary = create_summary(adata_subset_labelled, LABEL_COLUMN)
summary.to_csv(SUMMARY_PATH, index=False)

print("\nFinal Level 3 distribution:")
print(summary)

if "cell_type_level_1" in adata_subset_labelled.obs.columns:
    print("\nFinal Level 1 distribution:")
    print(adata_subset_labelled.obs["cell_type_level_1"].value_counts())

if "cell_type_level_2" in adata_subset_labelled.obs.columns:
    print("\nFinal Level 2 distribution:")
    print(adata_subset_labelled.obs["cell_type_level_2"].value_counts())


# Create unlabeled GUI input file
adata_subset_unlabeled = adata_subset_labelled.copy()

existing_label_columns = [
    col for col in LABEL_COLUMNS_TO_REMOVE
    if col in adata_subset_unlabeled.obs.columns
]

adata_subset_unlabeled.obs = adata_subset_unlabeled.obs.drop(
    columns=existing_label_columns
)

adata_subset_unlabeled.uns["gui_test_subset_info"]["label_removal_note"] = (
    "Ground-truth cell type labels were removed from adata.obs in this GUI input file."
)
adata_subset_unlabeled.uns["gui_test_subset_info"][
    "removed_label_columns"
] = existing_label_columns

# Remove label-focused metadata from the unlabeled file.
for key in ["label_hierarchy", "pancreas_data_structure"]:
    if key in adata_subset_unlabeled.uns:
        del adata_subset_unlabeled.uns[key]


# Save files
adata_subset_unlabeled.write_h5ad(OUTPUT_UNLABELED_PATH)
adata_subset_labelled.write_h5ad(OUTPUT_LABELLED_CONTROL_PATH)

print("\nSaved unlabeled GUI test set to:")
print(OUTPUT_UNLABELED_PATH)

print("\nSaved labelled control file to:")
print(OUTPUT_LABELLED_CONTROL_PATH)

print("\nSaved summary table to:")
print(SUMMARY_PATH)

print("\nRemaining obs columns in unlabeled GUI input:")
print(list(adata_subset_unlabeled.obs.columns))

print("\nObs columns in labelled control file:")
print(list(adata_subset_labelled.obs.columns))

print("\nFinal check:")
print(f"Unlabeled GUI input shape: {adata_subset_unlabeled.shape}")
print(f"Label columns removed: {existing_label_columns}")
print(f"Source used for GUI input: {RAW_ANNOTATED_PATH}")
