# 03_preprocess_train_pancreas.py

from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse


# Paths
INPUT_PATH = Path(
    r"C:\Users\ferid\Downloads\capstone_demo\pancreas_model\Tabula_Sapiens_Pancreas_annotated_with_levels.h5ad"
)

OUTPUT_DIR = INPUT_PATH.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "Tabula_Sapiens_Pancreas_preprocessed_for_training.h5ad"
HVG_CSV_PATH = OUTPUT_DIR / "pancreas_hvg_list.csv"
METADATA_PATH = OUTPUT_DIR / "pancreas_training_preprocessing_metadata.json"


# Parameters
MIN_COUNTS = 500
MIN_GENES = 200
N_TOP_GENES = 2000
LAYER_RAW_COUNTS = "counts"


def preprocess_for_training(
    adata: AnnData,
    min_counts: int = 500,
    min_genes: int = 200,
    target_sum: Optional[float] = None,
    n_top_genes: int = 2000,
    layer_raw_counts: str = "counts",
    copy: bool = True,
) -> AnnData:
    if copy:
        adata = adata.copy()

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Input AnnData is empty.")

    if adata.var_names.has_duplicates:
        raise ValueError("adata.var_names contains duplicate gene IDs.")

    required_label_columns = [
        "cell_type_original",
        "cell_type_standardized",
        "cell_type_level_1",
        "cell_type_level_2",
        "cell_type_level_3",
    ]

    for col in required_label_columns:
        if col not in adata.obs.columns:
            raise ValueError(f"Missing required label column: {col}")

    if "gene_symbol" not in adata.var.columns:
        raise ValueError("Missing required gene_symbol column in adata.var.")

    adata.obs_names_make_unique()

    adata.layers[layer_raw_counts] = adata.X.copy()

    sc.pp.calculate_qc_metrics(adata, inplace=True)

    n_cells_before_filtering = adata.n_obs
    n_genes_before_filtering = adata.n_vars

    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)

    sc.pp.filter_genes(adata, min_counts=1)

    n_cells_after_filtering = adata.n_obs
    n_genes_after_filtering = adata.n_vars

    if sparse.issparse(adata.X):
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel()
    else:
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel()

    if target_sum is None:
        target_sum = float(np.median(total_counts))

    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=n_top_genes,
        inplace=True,
    )

    adata.raw = adata

    hvg_mask = adata.var["highly_variable"].astype(bool)

    hvg_ensembl_ids = adata.var_names[hvg_mask].tolist()
    hvg_gene_symbols = adata.var.loc[hvg_mask, "gene_symbol"].astype(str).tolist()

    adata.uns["hvg_list"] = hvg_ensembl_ids
    adata.uns["hvg_order"] = hvg_ensembl_ids
    adata.uns["n_hvgs"] = int(len(hvg_ensembl_ids))
    adata.uns["hvg_gene_symbols"] = hvg_gene_symbols

    adata = adata[:, hvg_mask].copy()

    adata.uns["preprocessing"] = {
        "gene_id_used_for_hvg_and_alignment": "Ensembl ID in adata.var_names",
        "gene_symbol_column": "adata.var['gene_symbol']",
        "min_counts": int(min_counts),
        "min_genes": int(min_genes),
        "target_sum": float(target_sum),
        "n_top_genes": int(n_top_genes),
        "normalization": "library size normalization to median total counts per cell",
        "log_transform": "log1p",
        "hvg_method": "Scanpy highly_variable_genes(flavor='seurat')",
        "n_cells_before_filtering": int(n_cells_before_filtering),
        "n_cells_after_filtering": int(n_cells_after_filtering),
        "n_genes_before_filtering": int(n_genes_before_filtering),
        "n_genes_after_filtering_before_hvg": int(n_genes_after_filtering),
        "n_hvgs_final": int(adata.n_vars),
        "raw_counts_layer": layer_raw_counts,
    }

    return adata


def save_hvg_list(adata: AnnData, output_csv: str | Path) -> None:
    output_csv = Path(output_csv)

    hvg_df = pd.DataFrame({
        "gene_order": range(len(adata.var_names)),
        "ensembl_id": adata.var_names.astype(str),
        "gene_symbol": adata.var["gene_symbol"].astype(str).values,
    })

    hvg_df.to_csv(output_csv, index=False)


def save_label_distributions(adata: AnnData, output_dir: Path) -> None:
    label_columns = [
        "cell_type_level_1",
        "cell_type_level_2",
        "cell_type_level_3",
    ]

    for col in label_columns:
        counts = adata.obs[col].value_counts()
        counts.to_csv(output_dir / f"{col}_training_distribution.csv", header=["n_cells"])


# Load dataset
adata = sc.read_h5ad(INPUT_PATH)

print("Input dataset:")
print(adata)


# Preprocess
adata_pp = preprocess_for_training(
    adata,
    min_counts=MIN_COUNTS,
    min_genes=MIN_GENES,
    n_top_genes=N_TOP_GENES,
    layer_raw_counts=LAYER_RAW_COUNTS,
)


# Save outputs
adata_pp.write_h5ad(OUTPUT_PATH)
save_hvg_list(adata_pp, HVG_CSV_PATH)
save_label_distributions(adata_pp, OUTPUT_DIR)

metadata = {
    "input_dataset": str(INPUT_PATH),
    "output_dataset": str(OUTPUT_PATH),
    "hvg_csv": str(HVG_CSV_PATH),
    "n_cells": int(adata_pp.n_obs),
    "n_hvgs": int(adata_pp.n_vars),
    "label_columns": [
        "cell_type_level_1",
        "cell_type_level_2",
        "cell_type_level_3",
    ],
    "preprocessing": adata_pp.uns["preprocessing"],
}

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)


# Final checks
print("\nPreprocessing finished successfully.")
print(adata_pp)

print("\nLevel 1 distribution:")
print(adata_pp.obs["cell_type_level_1"].value_counts())

print("\nLevel 2 distribution:")
print(adata_pp.obs["cell_type_level_2"].value_counts())

print("\nLevel 3 distribution:")
print(adata_pp.obs["cell_type_level_3"].value_counts())

print(f"\nCells kept: {adata_pp.n_obs}")
print(f"HVGs kept: {adata_pp.n_vars}")

print("\nSaved preprocessed dataset to:")
print(OUTPUT_PATH)

print("\nSaved HVG list to:")
print(HVG_CSV_PATH)

print("\nSaved metadata to:")
print(METADATA_PATH)