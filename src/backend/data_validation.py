"""
data_validation.py
==================
Input data validation module for the Cell Type Prediction Platform.

Validates user-uploaded scRNA-seq datasets in .h5ad, .csv, and .mtx formats
before passing them to the preprocessing pipeline.

Checks performed:
- File format correctness (.h5ad, .csv, .mtx)
- Matrix dimensionality (must be 2-D, non-empty)
- Numerical validity (no NaN, no Inf, non-negative counts)
- Missing or invalid values
- Duplicate cell barcodes / gene IDs
- Minimum size requirements

Usage:
    from data_validation import validate_file, ValidationResult

    result = validate_file("path/to/data.h5ad")
    if result.is_valid:
        adata = result.data          # anndata.AnnData object, ready for preprocessing
    else:
        print(result.errors)        # list of human-readable error strings
        print(result.warnings)      # list of non-fatal warning strings
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Returned by every validate_* function."""

    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Populated on success
    data: Optional[object] = None          # anndata.AnnData
    n_cells: int = 0
    n_genes: int = 0
    file_format: str = ""

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = []
        if self.is_valid:
            lines.append(f"✅ Validation passed  |  {self.n_cells:,} cells  ×  {self.n_genes:,} genes  [{self.file_format}]")
        else:
            lines.append(f"❌ Validation failed  ({len(self.errors)} error(s))")
        for e in self.errors:
            lines.append(f"   ERROR   : {e}")
        for w in self.warnings:
            lines.append(f"   WARNING : {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Constants / thresholds
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".h5ad", ".csv", ".mtx"}

# Absolute minimums – datasets smaller than these are almost certainly corrupt
# or demo files that won't yield meaningful predictions.
MIN_CELLS = 10
MIN_GENES = 50

# A sparse matrix is expected for real scRNA-seq; warn if density is very high
# (might indicate a non-count / already-normalised matrix was uploaded).
HIGH_DENSITY_THRESHOLD = 0.8   # fraction of non-zero entries


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_file(filepath: str | Path) -> ValidationResult:
    """
    Validate a user-uploaded scRNA-seq file.

    Parameters
    ----------
    filepath : str or Path
        Path to the uploaded file (.h5ad, .csv, or .mtx).

    Returns
    -------
    ValidationResult
        .is_valid  → True if all checks pass
        .errors    → list of blocking error messages
        .warnings  → list of non-fatal advisory messages
        .data      → anndata.AnnData object (only set when is_valid is True)
        .n_cells   → number of cells
        .n_genes   → number of genes
    """
    path = Path(filepath)
    result = ValidationResult(is_valid=True)

    # ------------------------------------------------------------------
    # 1. File existence
    # ------------------------------------------------------------------
    if not path.exists():
        result.add_error(f"File not found: '{path}'")
        return result

    if not path.is_file():
        result.add_error(f"Path is not a file: '{path}'")
        return result

    # ------------------------------------------------------------------
    # 2. Format / extension check
    # ------------------------------------------------------------------
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        result.add_error(
            f"Unsupported file format '{ext}'. "
            f"Accepted formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
        return result

    result.file_format = ext.lstrip(".")

    # ------------------------------------------------------------------
    # 3. Non-zero file size
    # ------------------------------------------------------------------
    file_size = path.stat().st_size
    if file_size == 0:
        result.add_error("File is empty (0 bytes).")
        return result

    # ------------------------------------------------------------------
    # 4. Format-specific loading + checks
    # ------------------------------------------------------------------
    try:
        if ext == ".h5ad":
            _validate_h5ad(path, result)
        elif ext == ".csv":
            _validate_csv(path, result)
        elif ext == ".mtx":
            _validate_mtx(path, result)
    except Exception as exc:
        result.add_error(f"Unexpected error while reading file: {exc}")
        result.is_valid = False
        return result

    return result


# ---------------------------------------------------------------------------
# Format-specific validators
# ---------------------------------------------------------------------------

def _validate_h5ad(path: Path, result: ValidationResult) -> None:
    """Validate an AnnData .h5ad file."""
    try:
        import anndata as ad
    except ImportError:
        result.add_error(
            "Python package 'anndata' is not installed. "
            "Install it with: pip install anndata"
        )
        return

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adata = ad.read_h5ad(path)
    except Exception as exc:
        result.add_error(
            f"Could not read .h5ad file. "
            f"The file may be corrupt or not a valid AnnData object. "
            f"Details: {exc}"
        )
        return

    _check_anndata(adata, result)

    if result.is_valid:
        result.data = adata
        result.n_cells, result.n_genes = adata.shape


def _validate_csv(path: Path, result: ValidationResult) -> None:
    """
    Validate a CSV gene-expression matrix.

    Expected layout: rows = cells (or genes), columns = genes (or cells).
    The function accepts both orientations; if genes > cells it warns the user
    to transpose.
    """
    try:
        import pandas as pd
    except ImportError:
        result.add_error(
            "Python package 'pandas' is not installed. "
            "Install it with: pip install pandas"
        )
        return

    try:
        # Read only a small header first to detect delimiter / encoding issues.
        sample = pd.read_csv(path, index_col=0, nrows=5)
    except Exception as exc:
        result.add_error(
            f"Could not parse CSV file. "
            f"Ensure the file uses comma separators and UTF-8 encoding. "
            f"Details: {exc}"
        )
        return

    if sample.shape[1] == 0:
        result.add_error(
            "CSV file appears to have no data columns. "
            "Expected format: first column = cell/gene IDs, "
            "remaining columns = gene expression values."
        )
        return

    # Full read
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as exc:
        result.add_error(f"Failed to fully read CSV file: {exc}")
        return

    # Convert to numeric only
    non_numeric_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric_cols:
        result.add_error(
            f"{len(non_numeric_cols)} column(s) contain non-numeric data: "
            f"{non_numeric_cols[:5]}{'...' if len(non_numeric_cols) > 5 else ''}. "
            f"All expression values must be numeric."
        )
        return

    matrix = df.values
    row_ids = df.index.tolist()
    col_ids = df.columns.tolist()

    _check_matrix(matrix, row_ids, col_ids, result)

    if result.is_valid:
        # Build AnnData so downstream code works uniformly
        try:
            import anndata as ad
            import scipy.sparse as sp

            # Assume rows = cells, columns = genes
            adata = ad.AnnData(
                X=sp.csr_matrix(matrix.astype(np.float32)),
                obs=pd.DataFrame(index=row_ids),
                var=pd.DataFrame(index=col_ids),
            )
            result.data = adata
            result.n_cells, result.n_genes = adata.shape
        except ImportError:
            # anndata not available – store the raw DataFrame instead
            result.data = df
            result.n_cells, result.n_genes = df.shape
            result.add_warning(
                "Package 'anndata' is not installed; storing raw DataFrame. "
                "Install anndata for full pipeline compatibility."
            )


def _validate_mtx(path: Path, result: ValidationResult) -> None:
    """
    Validate a Matrix Market (.mtx) file.

    Expects the companion files barcodes.tsv / barcodes.tsv.gz and
    features.tsv / genes.tsv in the same directory (10x Genomics layout).
    """
    try:
        import scipy.io as sio
    except ImportError:
        result.add_error(
            "Python package 'scipy' is not installed. "
            "Install it with: pip install scipy"
        )
        return

    try:
        matrix = sio.mmread(path).tocsr()
    except Exception as exc:
        result.add_error(
            f"Could not read .mtx file. "
            f"Ensure the file is a valid Matrix Market format. "
            f"Details: {exc}"
        )
        return

    # Try to load barcodes and features from the same directory (10x layout)
    parent = path.parent
    barcode_file = _find_companion(parent, ["barcodes.tsv", "barcodes.tsv.gz"])
    feature_file = _find_companion(
        parent, ["features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz"]
    )

    import pandas as pd

    if barcode_file is None:
        result.add_warning(
            "No barcodes.tsv file found in the same directory. "
            "Cell barcodes will be auto-generated. "
            "For best results, place barcodes.tsv alongside the .mtx file."
        )
        barcodes = [f"cell_{i}" for i in range(matrix.shape[1])]
    else:
        try:
            barcodes = pd.read_csv(barcode_file, header=None, sep="\t")[0].tolist()
        except Exception as exc:
            result.add_error(f"Could not read barcodes file: {exc}")
            return

    if feature_file is None:
        result.add_warning(
            "No features.tsv / genes.tsv file found in the same directory. "
            "Gene IDs will be auto-generated."
        )
        genes = [f"gene_{i}" for i in range(matrix.shape[0])]
    else:
        try:
            feat_df = pd.read_csv(feature_file, header=None, sep="\t")
            genes = feat_df[0].tolist()  # first column = Ensembl ID or gene symbol
        except Exception as exc:
            result.add_error(f"Could not read features file: {exc}")
            return

    # MTX is stored (genes × cells) in 10x layout – transpose to (cells × genes)
    matrix_T = matrix.T

    _check_matrix(matrix_T.toarray(), barcodes, genes, result)

    if result.is_valid:
        try:
            import anndata as ad

            adata = ad.AnnData(
                X=matrix_T.astype(np.float32),
                obs=pd.DataFrame(index=barcodes),
                var=pd.DataFrame(index=genes),
            )
            result.data = adata
            result.n_cells, result.n_genes = adata.shape
        except ImportError:
            result.add_warning(
                "Package 'anndata' not installed; cannot build AnnData object. "
                "Install anndata for full pipeline compatibility."
            )


# ---------------------------------------------------------------------------
# Shared matrix checks
# ---------------------------------------------------------------------------

def _check_anndata(adata, result: ValidationResult) -> None:
    """Run all checks on an already-loaded AnnData object."""
    import numpy as np

    # --- Dimensionality ---
    if adata.X is None:
        result.add_error("AnnData object has no expression matrix (adata.X is None).")
        return

    if adata.n_obs == 0 or adata.n_vars == 0:
        result.add_error(
            f"Expression matrix is empty "
            f"({adata.n_obs} cells × {adata.n_vars} genes)."
        )
        return

    if adata.X.ndim != 2:
        result.add_error(
            f"Expression matrix must be 2-dimensional "
            f"(found {adata.X.ndim}-D array)."
        )
        return

    # --- Minimum size ---
    if adata.n_obs < MIN_CELLS:
        result.add_error(
            f"Dataset contains only {adata.n_obs} cell(s); "
            f"minimum required is {MIN_CELLS}. "
            f"Predictions are not meaningful on such small datasets."
        )

    if adata.n_vars < MIN_GENES:
        result.add_error(
            f"Dataset contains only {adata.n_vars} gene(s); "
            f"minimum required is {MIN_GENES}. "
            f"Ensure the full gene expression matrix was uploaded."
        )

    if not result.is_valid:
        return

    # --- Dense array extraction for numeric checks ---
    try:
        import scipy.sparse as sp
        X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    except Exception as exc:
        result.add_error(f"Could not convert expression matrix to array: {exc}")
        return

    _check_numeric(X, result)
    _check_duplicates(
        list(adata.obs_names), "cell barcode", result
    )
    _check_duplicates(
        list(adata.var_names), "gene ID", result
    )
    _check_density(X, result)


def _check_matrix(
    matrix: np.ndarray,
    row_ids: list,
    col_ids: list,
    result: ValidationResult,
) -> None:
    """Run all checks on a raw numpy matrix (cells × genes)."""
    n_cells, n_genes = matrix.shape

    if n_cells == 0 or n_genes == 0:
        result.add_error(
            f"Expression matrix is empty ({n_cells} cells × {n_genes} genes)."
        )
        return

    if matrix.ndim != 2:
        result.add_error(
            f"Expression matrix must be 2-dimensional "
            f"(found {matrix.ndim}-D array)."
        )
        return

    if n_cells < MIN_CELLS:
        result.add_error(
            f"Dataset contains only {n_cells} cell(s); "
            f"minimum required is {MIN_CELLS}."
        )

    if n_genes < MIN_GENES:
        result.add_error(
            f"Dataset contains only {n_genes} gene(s); "
            f"minimum required is {MIN_GENES}."
        )

    if not result.is_valid:
        return

    _check_numeric(matrix, result)
    _check_duplicates(row_ids, "cell barcode (row)", result)
    _check_duplicates(col_ids, "gene ID (column)", result)
    _check_density(matrix, result)


def _check_numeric(X: np.ndarray, result: ValidationResult) -> None:
    """Ensure matrix contains only finite, non-negative numbers."""
    nan_count = int(np.sum(np.isnan(X)))
    if nan_count > 0:
        result.add_error(
            f"Expression matrix contains {nan_count:,} NaN value(s). "
            f"NaN entries indicate missing data and must be resolved before upload. "
            f"Replace NaN with 0 for absent genes or remove affected cells/genes."
        )

    inf_count = int(np.sum(np.isinf(X)))
    if inf_count > 0:
        result.add_error(
            f"Expression matrix contains {inf_count:,} infinite value(s). "
            f"Infinite values are not valid gene expression counts."
        )

    neg_count = int(np.sum(X < 0))
    if neg_count > 0:
        result.add_error(
            f"Expression matrix contains {neg_count:,} negative value(s). "
            f"Raw gene expression counts must be non-negative integers. "
            f"If this is a pre-normalised matrix, ensure it is non-negative."
        )

    # Warn if values look like already-normalised floats rather than raw counts
    if result.is_valid:
        if X.dtype.kind == 'f':
            n_non_integer = int(np.sum(X != np.floor(X)))
            if n_non_integer > 0:
                fraction = n_non_integer / X.size
                if fraction > 0.01:   # more than 1 % non-integer values
                    result.add_warning(
                        f"{fraction*100:.1f}% of expression values are non-integer. "
                        f"The platform expects raw UMI count matrices. "
                        f"If this data is already normalised, preprocessing results may differ."
                    )


def _check_duplicates(ids: list, label: str, result: ValidationResult) -> None:
    """Detect and report duplicated identifiers."""
    seen = {}
    for idx, item in enumerate(ids):
        seen.setdefault(item, []).append(idx)

    duplicates = {k: v for k, v in seen.items() if len(v) > 1}
    if duplicates:
        examples = list(duplicates.items())[:5]
        example_str = "; ".join(
            f"'{k}' at rows {v}" for k, v in examples
        )
        result.add_error(
            f"Found {len(duplicates)} duplicate {label}(s). "
            f"All identifiers must be unique. "
            f"Examples: {example_str}"
            + ("  ..." if len(duplicates) > 5 else "")
        )


def _check_density(X: np.ndarray, result: ValidationResult) -> None:
    """Warn if the matrix is unexpectedly dense (may not be raw counts)."""
    total = X.size
    if total == 0:
        return
    non_zero = int(np.count_nonzero(X))
    density = non_zero / total
    if density > HIGH_DENSITY_THRESHOLD:
        result.add_warning(
            f"Expression matrix density is {density*100:.1f}% "
            f"(fraction of non-zero entries). "
            f"Raw scRNA-seq count matrices are typically sparse (<50% non-zero). "
            f"This may indicate that the data has already been normalised or transformed."
        )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _find_companion(directory: Path, candidates: list) -> Optional[Path]:
    """Return the first existing file from a list of candidate names."""
    for name in candidates:
        p = directory / name
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def is_valid_h5ad(filepath: str | Path) -> bool:
    """Quick boolean check for .h5ad files."""
    return validate_file(filepath).is_valid


def get_validation_errors(filepath: str | Path) -> List[str]:
    """Return only the list of error strings for a given file."""
    return validate_file(filepath).errors


# ---------------------------------------------------------------------------
# CLI usage: python data_validation.py <file>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_validation.py <path_to_file>")
        sys.exit(1)

    target = sys.argv[1]
    res = validate_file(target)
    print(res.summary())
    sys.exit(0 if res.is_valid else 1)
