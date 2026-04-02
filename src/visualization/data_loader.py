"""
data_loader.py
==============
File loading module for the Cell Type Prediction Platform.

Accepts user-uploaded scRNA-seq datasets in .h5ad, .csv, and .mtx formats
and converts them all into a standard AnnData object for downstream use.

Supported formats:
    - .h5ad  : AnnData HDF5 file (Scanpy output)
    - .csv   : Dense gene expression matrix (cells × genes)
    - .mtx   : Matrix Market sparse format (10x Genomics layout)
                └── expects barcodes.tsv + features.tsv in the same directory

Usage:
    from data_loader import load_file, LoadResult

    result = load_file("path/to/data.h5ad")
    if result.success:
        adata = result.adata      # anndata.AnnData, ready for validation
    else:
        print(result.error)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadResult:
    """Returned by load_file()."""
    success: bool = False
    adata: Optional[object] = None   # anndata.AnnData on success
    file_format: str = ""            # "h5ad" | "csv" | "mtx"
    n_cells: int = 0
    n_genes: int = 0
    error: str = ""                  # set on failure

    def __repr__(self) -> str:
        if self.success:
            return (
                f"LoadResult(success=True, format={self.file_format!r}, "
                f"n_cells={self.n_cells:,}, n_genes={self.n_genes:,})"
            )
        return f"LoadResult(success=False, error={self.error!r})"


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def load_file(filepath: str | Path) -> LoadResult:
    """
    Load a scRNA-seq dataset and return a standard AnnData object.

    Parameters
    ----------
    filepath : str or Path
        Path to the uploaded file (.h5ad, .csv, or .mtx).

    Returns
    -------
    LoadResult
        .success     → True if loading succeeded
        .adata       → anndata.AnnData object (cells × genes)
        .file_format → "h5ad" | "csv" | "mtx"
        .n_cells     → number of cells (rows)
        .n_genes     → number of genes (columns)
        .error       → human-readable error message on failure
    """
    path = Path(filepath)

    # ── existence ─────────────────────────────────────────────────────────────
    if not path.exists():
        return LoadResult(error=f"File not found: '{path}'")
    if not path.is_file():
        return LoadResult(error=f"Path is not a file: '{path}'")
    if path.stat().st_size == 0:
        return LoadResult(error="File is empty (0 bytes).")

    ext = path.suffix.lower()

    if ext == ".h5ad":
        return _load_h5ad(path)
    elif ext == ".csv":
        return _load_csv(path)
    elif ext == ".mtx":
        return _load_mtx(path)
    else:
        return LoadResult(
            error=(
                f"Unsupported format '{ext}'. "
                f"Accepted: .h5ad, .csv, .mtx"
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# Format-specific loaders
# ─────────────────────────────────────────────────────────────────────────────

def _load_h5ad(path: Path) -> LoadResult:
    """Load an AnnData .h5ad file."""
    try:
        import anndata as ad
    except ImportError:
        return LoadResult(
            error="Package 'anndata' is not installed. Run: pip install anndata"
        )

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata = ad.read_h5ad(path)
    except Exception as exc:
        return LoadResult(
            error=(
                f"Could not read .h5ad file — file may be corrupt or "
                f"not a valid AnnData object. Details: {exc}"
            )
        )

    # Ensure X is a CSR sparse matrix for consistency
    adata = _to_csr(adata)

    return LoadResult(
        success=True,
        adata=adata,
        file_format="h5ad",
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
    )


def _load_csv(path: Path) -> LoadResult:
    """
    Load a CSV gene-expression matrix.

    Expected layout:
        - Row 0    : header (gene IDs / names)
        - Column 0 : cell barcodes
        - Values   : numeric expression counts
    """
    try:
        import pandas as pd
    except ImportError:
        return LoadResult(
            error="Package 'pandas' is not installed. Run: pip install pandas"
        )

    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as exc:
        return LoadResult(
            error=(
                f"Could not parse CSV file. "
                f"Ensure comma separators and UTF-8 encoding. Details: {exc}"
            )
        )

    if df.shape[0] == 0 or df.shape[1] == 0:
        return LoadResult(
            error=(
                f"CSV matrix is empty ({df.shape[0]} rows × {df.shape[1]} cols). "
                f"Expected: first column = cell barcodes, header = gene IDs."
            )
        )

    # Coerce to numeric — non-numeric columns become NaN
    df = df.apply(pd.to_numeric, errors="coerce")

    non_num = int(df.isnull().sum().sum())
    if non_num > 0 and non_num == df.size:
        return LoadResult(
            error="All values in the CSV are non-numeric. Check the file format."
        )

    try:
        import anndata as ad
        import scipy.sparse as sp

        adata = ad.AnnData(
            X=sp.csr_matrix(df.values.astype(np.float32)),
            obs=pd.DataFrame(index=df.index.astype(str)),
            var=pd.DataFrame(index=df.columns.astype(str)),
        )
    except ImportError:
        return LoadResult(
            error="Package 'anndata' is not installed. Run: pip install anndata"
        )
    except Exception as exc:
        return LoadResult(error=f"Failed to build AnnData from CSV: {exc}")

    return LoadResult(
        success=True,
        adata=adata,
        file_format="csv",
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
    )


def _load_mtx(path: Path) -> LoadResult:
    """
    Load a 10x Genomics Matrix Market dataset.

    Directory layout expected:
        matrix.mtx   (or any .mtx file)
        barcodes.tsv (or barcodes.tsv.gz)
        features.tsv / genes.tsv (or .gz variants)

    The .mtx file is stored as genes × cells; this function transposes it
    to the standard cells × genes orientation.
    """
    try:
        import scipy.io as sio
        import pandas as pd
    except ImportError:
        return LoadResult(
            error="Package 'scipy' is not installed. Run: pip install scipy"
        )

    try:
        matrix = sio.mmread(path).tocsr()   # genes × cells
    except Exception as exc:
        return LoadResult(
            error=f"Could not read .mtx file — ensure it is valid Matrix Market format. Details: {exc}"
        )

    parent = path.parent

    # ── barcodes ──────────────────────────────────────────────────────────────
    barcode_path = _find_file(
        parent, ["barcodes.tsv", "barcodes.tsv.gz"]
    )
    if barcode_path is None:
        # Auto-generate barcodes rather than hard-failing
        barcodes = [f"cell_{i}" for i in range(matrix.shape[1])]
    else:
        try:
            barcodes = (
                pd.read_csv(barcode_path, header=None, sep="\t")[0]
                .astype(str)
                .tolist()
            )
        except Exception as exc:
            return LoadResult(error=f"Could not read barcodes file: {exc}")

    # ── features / genes ──────────────────────────────────────────────────────
    feature_path = _find_file(
        parent,
        ["features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"],
    )
    if feature_path is None:
        genes = [f"gene_{i}" for i in range(matrix.shape[0])]
    else:
        try:
            feat_df = pd.read_csv(feature_path, header=None, sep="\t")
            # Column 0 = Ensembl ID, column 1 = gene symbol (if present)
            genes = feat_df[0].astype(str).tolist()
        except Exception as exc:
            return LoadResult(error=f"Could not read features/genes file: {exc}")

    # ── dimension check ───────────────────────────────────────────────────────
    if matrix.shape[1] != len(barcodes):
        return LoadResult(
            error=(
                f"Barcode count ({len(barcodes)}) does not match "
                f"number of columns in .mtx ({matrix.shape[1]}). "
                f"Ensure barcodes.tsv matches the matrix."
            )
        )
    if matrix.shape[0] != len(genes):
        return LoadResult(
            error=(
                f"Gene count ({len(genes)}) does not match "
                f"number of rows in .mtx ({matrix.shape[0]}). "
                f"Ensure features.tsv matches the matrix."
            )
        )

    # ── transpose to cells × genes ────────────────────────────────────────────
    matrix_T = matrix.T.tocsr().astype(np.float32)   # cells × genes

    try:
        import anndata as ad

        adata = ad.AnnData(
            X=matrix_T,
            obs=pd.DataFrame(index=barcodes),
            var=pd.DataFrame(index=genes),
        )
    except ImportError:
        return LoadResult(
            error="Package 'anndata' is not installed. Run: pip install anndata"
        )
    except Exception as exc:
        return LoadResult(error=f"Failed to build AnnData from .mtx: {exc}")

    return LoadResult(
        success=True,
        adata=adata,
        file_format="mtx",
        n_cells=adata.n_obs,
        n_genes=adata.n_vars,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_file(directory: Path, candidates: list[str]) -> Optional[Path]:
    """Return the first matching file from a list of candidate names."""
    for name in candidates:
        p = directory / name
        if p.exists():
            return p
    return None


def _to_csr(adata):
    """Convert adata.X to CSR sparse format if not already."""
    import scipy.sparse as sp
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X.astype(np.float32))
    elif not isinstance(adata.X, sp.csr_matrix):
        adata.X = adata.X.tocsr()
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# CLI:  python data_loader.py <file>
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <path_to_file>")
        sys.exit(1)

    res = load_file(sys.argv[1])
    print(res)
    sys.exit(0 if res.success else 1)
