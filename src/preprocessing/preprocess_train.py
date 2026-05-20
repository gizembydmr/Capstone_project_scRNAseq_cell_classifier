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
    """
    Preprocess a single-cell RNA-seq AnnData object for model training.

    Parameters
    ----------
    adata : AnnData
        Input dataset. adata.X is expected to contain raw count values
        (integer-like UMI/read counts), with cells in rows and genes in columns.

    min_counts : int, default=500
        Minimum total count required for a cell to be kept.
        This is a light quality filter. Cells below this threshold often have
        too little captured RNA and may represent very low-quality droplets,
        broken cells, or nearly empty barcodes.

    min_genes : int, default=200
        Minimum number of detected genes required for a cell to be kept.
        A gene is considered detected in a cell if its count is > 0.
        Cells with very few detected genes usually contain too little biological
        information for reliable downstream analysis.

    target_sum : float or None, default=None
        Target total count after library-size normalization.
        If None, we use the median total count across cells, following the logic
        of the PBMC68k R workflow. This keeps normalized values on a realistic
        scale from the dataset itself instead of forcing an arbitrary constant.

    n_top_genes : int, default=2000
        Number of highly variable genes (HVGs) to keep.
        2000 is a common practical default in scRNA-seq because it captures a
        strong biological signal while reducing noise and dimensionality.
        This can later be tuned.

    layer_raw_counts : str, default="counts"
        Name of the AnnData layer where the original raw counts will be stored
        before normalization/log transformation.

    copy : bool, default=True
        If True, work on a copy of the input AnnData and leave the original
        object unchanged.

    Important assumption for this project
    -------------------------------------
    - adata.var_names must contain Ensembl IDs (unique gene identifiers)
    - adata.var["gene_symbol"] may contain gene symbols for readability
    - HVG selection and downstream gene alignment should be based on Ensembl IDs,
      not gene symbols, because gene symbols are not always unique

    Returns
    -------
    AnnData
        Preprocessed AnnData object containing only HVGs in adata.X.
        The full normalized+log-transformed matrix before HVG subsetting is
        stored in adata.raw, and preprocessing metadata are stored in adata.uns.
    """

    if copy:
        adata = adata.copy()

    # Safety check: an empty dataset cannot be processed.
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise ValueError("Input AnnData is empty.")

    # We EXPECT var_names to already contain unique Ensembl IDs.
    # In your dataset design, Ensembl IDs are the main gene identifiers and
    # gene symbols are stored separately only for readability.
    #
    # For that reason, unlike obs_names, we do NOT silently modify var_names.
    # If duplicate gene IDs exist here, that indicates a real upstream data
    # problem and should be fixed explicitly, not hidden by renaming.
    if adata.var_names.has_duplicates:
        raise ValueError(
            "adata.var_names contains duplicate gene IDs. "
            "For this project, var_names should be unique Ensembl IDs."
        )

    # obs_names are cell IDs / barcodes.
    # Here, making names unique is generally safe because this step does NOT
    # delete any cells. If duplicate row names exist, AnnData simply renames
    # them by appending suffixes such as:
    #   cellA, cellA-1, cellA-2
    # This prevents indexing ambiguity in downstream steps.
    adata.obs_names_make_unique()

    # Save raw counts before any normalization or transformation.
    # This is important because later steps overwrite adata.X.
    adata.layers[layer_raw_counts] = adata.X.copy()

    # Compute standard QC metrics per cell and per gene.
    # Examples added by Scanpy include:
    # - total_counts: total counts in each cell
    # - n_genes_by_counts: number of detected genes in each cell
    # - total counts per gene / number of cells expressing each gene
    #
    # We compute them so the dataset contains transparent quality information.
    # Thresholds are still chosen by us, but these metrics are what those
    # filtering decisions are based on.
    sc.pp.calculate_qc_metrics(adata, inplace=True)

    # Light cell filtering.
    # We remove cells with extremely low RNA content or very few detected genes.
    # These cells are usually less informative and can add noise to training.
    sc.pp.filter_cells(adata, min_counts=min_counts)
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # Remove genes with no signal in the kept cells.
    # Such genes cannot help the model because they are zero everywhere.
    sc.pp.filter_genes(adata, min_counts=1)

    # Compute each cell's library size = total counts across all genes.
    # In sparse matrices, sum(axis=1) returns a matrix-like object, so we convert
    # it to a 1D NumPy array for easier use.
    if sparse.issparse(adata.X):
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel()
    else:
        total_counts = np.asarray(adata.X.sum(axis=1)).ravel()

    # If target_sum is not given, use the median library size of this dataset.
    # This follows the PBMC68k R script's logic:
    #   normalized_cell = raw_cell / (cell_total / median_total)
    # So cells with larger depth are scaled down and cells with smaller depth are
    # scaled up, bringing all cells to the same typical sequencing depth.
    if target_sum is None:
        target_sum = float(np.median(total_counts))

    # Normalize each cell for sequencing depth (library size normalization).
    # After this step, each cell has approximately the same total count.
    # This reduces technical variation caused by some cells being sequenced more
    # deeply than others.
    sc.pp.normalize_total(adata, target_sum=target_sum)

    # Log-transform the normalized values.
    # log1p means log(1 + x).
    # Why we do this:
    # 1) compresses the very large range of expression values,
    # 2) reduces the dominance of highly expressed genes,
    # 3) makes the data more stable for variance-based gene selection and later ML.
    # In the PBMC68k workflow, the explicit normalization-by-UMI step itself does
    # not log-transform immediately, but they DO use log(1 + x) later during PCA
    # preparation (.do_propack). Here we apply log1p already at preprocessing time
    # because it is a very standard and reusable step for a platform pipeline.
    sc.pp.log1p(adata)

    # Identify highly variable genes (HVGs).
    # HVGs are genes whose expression varies strongly across cells relative to
    # genes with similar average expression. These genes often carry the most
    # useful biological information for distinguishing cell types.
    #
    # Important for this project:
    # HVG identity is tracked using Ensembl IDs because adata.var_names is
    # expected to contain Ensembl IDs, not gene symbols.
    sc.pp.highly_variable_genes(
        adata,
        flavor="seurat",
        n_top_genes=n_top_genes,
        inplace=True,
    )

    # Save the full normalized/log-transformed matrix before restricting to HVGs.
    # adata.raw is a standard AnnData slot used to preserve a broader feature set.
    adata.raw = adata

    # Save HVG information for later inference-time gene alignment.
    # Zeynep can use these saved fields to align an input dataset to exactly the
    # same training gene set and the same gene order.
    #
    # Primary identifier: Ensembl ID
    # Optional readable label: gene_symbol
    hvg_mask = adata.var["highly_variable"].astype(bool)

    hvg_ensembl_ids = adata.var_names[hvg_mask].tolist()

    if "gene_symbol" in adata.var.columns:
        hvg_gene_symbols = adata.var.loc[hvg_mask, "gene_symbol"].astype(str).tolist()
    else:
        hvg_gene_symbols = [None] * len(hvg_ensembl_ids)

    adata.uns["hvg_list"] = hvg_ensembl_ids
    adata.uns["hvg_order"] = hvg_ensembl_ids  # same content, explicit name for clarity
    adata.uns["n_hvgs"] = int(len(hvg_ensembl_ids))
    adata.uns["hvg_gene_symbols"] = hvg_gene_symbols

    # Keep only HVGs in adata.X for downstream model training.
    adata = adata[:, hvg_mask].copy()

    # Save preprocessing settings inside the .h5ad file for reproducibility.
    # adata.uns is an unstructured metadata dictionary in AnnData.
    # When you save the object to .h5ad, these fields are saved with it.
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
    """
    Save the final HVG list and exact order to a CSV file.

    Output columns
    --------------
    gene_order : exact feature order used by the training matrix
    ensembl_id : primary identifier used for alignment
    gene_symbol : optional readable symbol, if available
    """
    output_csv = Path(output_csv)

    if "gene_symbol" in adata.var.columns:
        gene_symbol = adata.var["gene_symbol"].astype(str).values
    else:
        gene_symbol = [None] * adata.n_vars

    hvg_df = pd.DataFrame(
        {
            "gene_order": range(len(adata.var_names)),
            "ensembl_id": adata.var_names,
            "gene_symbol": gene_symbol,
        }
    )
    hvg_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    # This block runs only when this file is executed directly, for example:
    # python preprocess_scRNA_train.py
    # It does NOT run when the function is imported from another script.

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

    # Save preprocessed training-ready dataset.
    adata_pp.write(output_path)

    # Save the HVG list and exact gene order for inference preprocessing.
    save_hvg_list(adata_pp, hvg_csv_path)

    print("Preprocessing finished successfully.")
    print(adata_pp)
    print(f"Cells kept: {adata_pp.n_obs}")
    print(f"HVGs kept: {adata_pp.n_vars}")
    print(f"Saved preprocessed dataset to: {output_path}")
    print(f"Saved HVG list to: {hvg_csv_path}")
