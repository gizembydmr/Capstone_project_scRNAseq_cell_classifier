from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
from anndata import AnnData
from scipy import sparse


def preprocess_for_inference(
    adata: AnnData,
    min_counts: int = 500,
    min_genes: int = 200,
    target_sum: Optional[float] = None,
    layer_raw_counts: str = "counts",
    copy: bool = True,
) -> AnnData:
    """
    Preprocess a single-cell RNA-seq AnnData object for model inference.

    This follows the same core logic as preprocess_train.py:
    - store raw counts
    - compute QC metrics
    - filter weak cells
    - remove zero-count genes
    - normalize for sequencing depth
    - log1p transform

    Important difference from training:
    - NO HVG selection is performed here
    - gene selection is enforced later via gene_alignment.py
    """

    if copy:
        adata = adata.copy()

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Input AnnData is empty.")

    if adata.var_names.has_duplicates:
        raise ValueError(
            "adata.var_names contains duplicate gene IDs. "
            "For this project, var_names should be unique Ensembl IDs."
        )

    adata.obs_names_make_unique()

    # Save raw counts
    adata.layers[layer_raw_counts] = adata.X.copy()

    # Compute QC metrics
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    # Filter weak cells
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)

    if adata.n_obs == 0:
        raise ValueError(
            "All cells were filtered out. Try lowering min_counts or min_genes."
        )

    # Remove genes with zero signal in the retained cells
    sc.pp.filter_genes(adata, min_counts=1)

    if adata.n_vars == 0:
        raise ValueError("All genes were removed after filtering.")

    # Compute library size
    if sparse.issparse(adata.X):
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel()
    else:
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel()

    # Match training logic: use median library size if not provided
    if target_sum is None:
        target_sum = float(np.median(total_counts))

    if target_sum <= 0:
        raise ValueError("Computed target_sum is <= 0. Cannot normalize.")

    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # Save full normalized/log-transformed matrix
    adata.raw = adata

    adata.uns["preprocessing_inference"] = {
        "gene_id_used_for_alignment": "Ensembl ID in adata.var_names",
        "min_counts": int(min_counts),
        "min_genes": int(min_genes),
        "target_sum": float(target_sum),
        "normalization": "library size normalization to median total counts per cell",
        "log_transform": "log1p",
        "note": "No HVG selection is performed during inference preprocessing.",
    }

    return adata


if __name__ == "__main__":
    input_path = Path("pbmc68k_annotated_with_levels.h5ad")
    output_path = Path("outputs/pbmc68k_preprocessed_for_inference.h5ad")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(input_path)

    adata_pp = preprocess_for_inference(
        adata,
        min_counts=500,
        min_genes=200,
    )

    adata_pp.write(output_path)

    print("Inference preprocessing finished successfully.")
    print(adata_pp)
    print(f"Cells kept: {adata_pp.n_obs}")
    print(f"Genes kept after filtering: {adata_pp.n_vars}")
    print(f"Saved preprocessed inference dataset to: {output_path}")