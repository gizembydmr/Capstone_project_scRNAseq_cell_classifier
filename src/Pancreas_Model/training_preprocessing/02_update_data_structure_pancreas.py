# 02_update_data_structure_pancreas.py

from pathlib import Path
import json

import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData


# Paths
DATA_PATH = Path(
    r"C:\Users\ferid\Downloads\capstone_demo\pancreas_model\Tabula_Sapiens_Pancreas_original.h5ad"
)

OUTPUT_DIR = DATA_PATH.parent
REPORT_DIR = OUTPUT_DIR / "pancreas_data_structure_outputs"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "Tabula_Sapiens_Pancreas_annotated_with_levels.h5ad"


# Parameters
MIN_CELLS_LEVEL3 = 20

COUNT_LAYER = "decontXcounts"
GENE_SYMBOL_COLUMN = "feature_name"

LEVEL1_SOURCE = "compartment"
LEVEL2_SOURCE = "broad_cell_class"
LEVEL3_SOURCE = "cell_type"


def plot_label_distribution(adata, column, output_dir, suffix):
    counts = adata.obs[column].value_counts()

    csv_path = output_dir / f"{column}_distribution_{suffix}.csv"
    counts.to_csv(csv_path, header=["n_cells"])

    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Number of cells")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    plot_path = output_dir / f"{column}_distribution_{suffix}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    return csv_path, plot_path


# Load dataset
adata_source = sc.read_h5ad(DATA_PATH)

print("Original dataset:")
print(adata_source)


# Checks
required_obs_columns = [
    LEVEL1_SOURCE,
    LEVEL2_SOURCE,
    LEVEL3_SOURCE,
]

for col in required_obs_columns:
    if col not in adata_source.obs.columns:
        raise ValueError(f"Missing required obs column: {col}")

if COUNT_LAYER not in adata_source.layers:
    raise ValueError(f"Missing required count layer: {COUNT_LAYER}")

if GENE_SYMBOL_COLUMN not in adata_source.var.columns:
    raise ValueError(f"Missing required gene symbol column: {GENE_SYMBOL_COLUMN}")

if adata_source.var_names.has_duplicates:
    raise ValueError("adata.var_names contains duplicate gene IDs.")


# Create clean AnnData
adata = AnnData(
    X=adata_source.layers[COUNT_LAYER].copy(),
    obs=adata_source.obs.copy(),
    var=adata_source.var.copy(),
)

adata.obs_names = adata_source.obs_names.copy()
adata.var_names = adata_source.var_names.copy()

adata.obs_names_make_unique()


# Gene metadata
adata.var["gene_symbol"] = adata.var[GENE_SYMBOL_COLUMN].astype(str)

if "ensembl_id" not in adata.var.columns:
    adata.var["ensembl_id"] = adata.var_names.astype(str)


# Cell metadata
adata.obs["barcode"] = adata.obs_names.astype(str)

adata.obs["cell_type_original"] = adata.obs[LEVEL3_SOURCE].astype(str)
adata.obs["cell_type_standardized"] = adata.obs[LEVEL3_SOURCE].astype(str)

adata.obs["cell_type_level_1"] = adata.obs[LEVEL1_SOURCE].astype(str)
adata.obs["cell_type_level_2"] = adata.obs[LEVEL2_SOURCE].astype(str)
adata.obs["cell_type_level_3"] = adata.obs[LEVEL3_SOURCE].astype(str)


# Label distributions before filtering
plot_label_distribution(adata, "cell_type_level_1", REPORT_DIR, "before_filtering")
plot_label_distribution(adata, "cell_type_level_2", REPORT_DIR, "before_filtering")
plot_label_distribution(adata, "cell_type_level_3", REPORT_DIR, "before_filtering")


# Rare label filtering
level3_counts_before = adata.obs["cell_type_level_3"].value_counts()

rare_level3_labels = level3_counts_before[
    level3_counts_before < MIN_CELLS_LEVEL3
].index.tolist()

rare_labels_df = pd.DataFrame({
    "cell_type_level_3": rare_level3_labels,
    "n_cells": [int(level3_counts_before[label]) for label in rare_level3_labels],
})

rare_labels_path = REPORT_DIR / "removed_rare_level3_labels.csv"
rare_labels_df.to_csv(rare_labels_path, index=False)

adata = adata[~adata.obs["cell_type_level_3"].isin(rare_level3_labels)].copy()


# Label columns
label_columns = [
    "cell_type_original",
    "cell_type_standardized",
    "cell_type_level_1",
    "cell_type_level_2",
    "cell_type_level_3",
]

for col in label_columns:
    adata.obs[col] = adata.obs[col].astype("category")


# Label distributions after filtering
plot_label_distribution(adata, "cell_type_level_1", REPORT_DIR, "after_filtering")
plot_label_distribution(adata, "cell_type_level_2", REPORT_DIR, "after_filtering")
plot_label_distribution(adata, "cell_type_level_3", REPORT_DIR, "after_filtering")


# Metadata
summary = {
    "input_dataset": str(DATA_PATH),
    "output_dataset": str(OUTPUT_PATH),
    "n_cells_before_filtering": int(level3_counts_before.sum()),
    "n_cells_after_filtering": int(adata.n_obs),
    "n_genes": int(adata.n_vars),
    "count_source": f"adata.layers['{COUNT_LAYER}']",
    "gene_symbol_source": f"adata.var['{GENE_SYMBOL_COLUMN}']",
    "level_1_source": LEVEL1_SOURCE,
    "level_2_source": LEVEL2_SOURCE,
    "level_3_source": LEVEL3_SOURCE,
    "min_cells_level3": int(MIN_CELLS_LEVEL3),
    "n_removed_level3_labels": int(len(rare_level3_labels)),
    "removed_level3_labels": rare_level3_labels,
    "level_1_classes": sorted(adata.obs["cell_type_level_1"].astype(str).unique().tolist()),
    "level_2_classes": sorted(adata.obs["cell_type_level_2"].astype(str).unique().tolist()),
    "level_3_classes": sorted(adata.obs["cell_type_level_3"].astype(str).unique().tolist()),
}

adata.uns["dataset_metadata"] = {
    "dataset_name": "Tabula Sapiens - Pancreas",
    "source": "CZ CELLxGENE Discover",
    "organism": "Homo sapiens",
    "tissue": "pancreas",
}

adata.uns["label_hierarchy"] = {
    "cell_type_level_1": LEVEL1_SOURCE,
    "cell_type_level_2": LEVEL2_SOURCE,
    "cell_type_level_3": LEVEL3_SOURCE,
}

adata.uns["pancreas_data_structure"] = summary

summary_path = REPORT_DIR / "pancreas_data_structure_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)


# Final checks
print("\nPrepared dataset:")
print(adata)

print("\nLevel 1 distribution:")
print(adata.obs["cell_type_level_1"].value_counts())

print("\nLevel 2 distribution:")
print(adata.obs["cell_type_level_2"].value_counts())

print("\nLevel 3 distribution:")
print(adata.obs["cell_type_level_3"].value_counts())

print("\nRemoved rare Level 3 labels:")
print(rare_labels_df)


# Save
adata.write_h5ad(OUTPUT_PATH)

print("\nSaved structured dataset to:")
print(OUTPUT_PATH)

print("\nSaved structure outputs to:")
print(REPORT_DIR)