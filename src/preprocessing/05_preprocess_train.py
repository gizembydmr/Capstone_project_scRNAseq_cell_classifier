from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse


def preprocess_for_training(
    adata: AnnData,
    min_counts: int = 500,
    min_genes: int = 200,
    target_sum: Optional[float] = None,
    n_top_genes: int = 2000,
    layer_raw_counts: str = "counts",
    copy: bool = True,
) -> AnnData:
    """Preprocess a single-cell RNA-seq AnnData object for model training."""

    if copy:
        adata = adata.copy()

    # Check that the dataset is not empty.
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Input AnnData is empty.")

    # Require unique Ensembl IDs in var_names for reliable gene alignment.
    if adata.var_names.has_duplicates:
        raise ValueError(
            "adata.var_names contains duplicate gene IDs. "
            "For this project, var_names should be unique Ensembl IDs."
        )

    # Make cell/barcode names unique to avoid indexing ambiguity.
    adata.obs_names_make_unique()

    # Store original raw counts before normalization and transformation.
    adata.layers[layer_raw_counts] = adata.X.copy()

    # Compute basic cell and gene quality-control metrics.
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    # Remove low-quality cells with very low counts or detected genes.
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # Remove genes with no remaining signal.
    sc.pp.filter_genes(adata, min_counts=1)

    # Compute total counts per cell.
    if sparse.issparse(adata.X):
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel()
    else:
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel()

    # Use the median library size as the normalization target if not provided.
    if target_sum is None:
        target_sum = float(np.median(total_counts))

    # Normalize cells to the selected library-size target.
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # Apply log1p transformation to stabilize expression values.
    sc.pp.log1p(adata)

    # Select highly variable genes for model training.
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=n_top_genes,
        inplace=True,
    )

    # Preserve the full normalized/log-transformed gene space.
    adata.raw = adata

    # Extract the selected HVG identifiers and optional gene symbols.
    hvg_mask = adata.var["highly_variable"].astype(bool)

    hvg_ensembl_ids = adata.var_names[hvg_mask].tolist()

    if "gene_symbol" in adata.var.columns:
        hvg_gene_symbols = adata.var.loc[hvg_mask, "gene_symbol"].astype(str).tolist()
    else:
        hvg_gene_symbols = [None] * len(hvg_ensembl_ids)

    # Store HVG metadata for inference-time gene alignment.
    adata.uns["hvg_list"] = hvg_ensembl_ids
    adata.uns["hvg_order"] = hvg_ensembl_ids  # Same content, explicit name for clarity.
    adata.uns["n_hvgs"] = int(len(hvg_ensembl_ids))
    adata.uns["hvg_gene_symbols"] = hvg_gene_symbols

    # Keep only HVGs in the active expression matrix.
    adata = adata[:, hvg_mask].copy()

    # Store preprocessing settings for reproducibility.
    adata.uns["preprocessing"] = {
        "gene_id_used_for_hvg_and_alignment": "Ensembl ID in adata.var_names",
        "gene_symbol_column": "adata.var['gene_symbol']" if "gene_symbol" in adata.var.columns else None,
        "min_counts": int(min_counts),
        "min_genes": int(min_genes),
        "target_sum": float(target_sum),
        "n_top_genes": int(n_top_genes),
        "normalization": "library size normalization to median total counts per cell",
        "log_transform": "log1p",
        "hvg_method": "Scanpy highly_variable_genes(flavor='seurat')",
    }

    return adata


def save_hvg_list(adata: AnnData, output_csv: str | Path) -> None:
    """Save the final HVG list and exact gene order to a CSV file."""
    output_csv = Path(output_csv)

    # Use gene symbols if available; otherwise store empty values.
    if "gene_symbol" in adata.var.columns:
        gene_symbol = adata.var["gene_symbol"].astype(str).values
    else:
        gene_symbol = [None] * adata.n_vars

    # Create a table containing the exact feature order used for training.
    hvg_df = pd.DataFrame(
        {
            "gene_order": range(len(adata.var_names)),
            "ensembl_id": adata.var_names,
            "gene_symbol": gene_symbol,
        }
    )
    hvg_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    # Run the training preprocessing pipeline when the script is executed directly.
    input_path = Path("pbmc68k_annotated_with_levels.h5ad")
    output_path = Path("outputs/pbmc68k_preprocessed_for_training.h5ad")
    hvg_csv_path = Path("outputs/pbmc68k_hvg_list.csv")

    adata = sc.read_h5ad(input_path)

    adata_pp = preprocess_for_training(
        adata,
        min_counts=500,
        min_genes=200,
        n_top_genes=2000,
    )

    # Save the preprocessed training-ready dataset.
    adata_pp.write(output_path)

    # Save the selected HVG list and order.
    save_hvg_list(adata_pp, hvg_csv_path)

    print("Preprocessing finished successfully.")
    print(adata_pp)
    print(f"Cells kept: {adata_pp.n_obs}")
    print(f"HVGs kept: {adata_pp.n_vars}")
    print(f"Saved preprocessed dataset to: {output_path}")
    print(f"Saved HVG list to: {hvg_csv_path}")
