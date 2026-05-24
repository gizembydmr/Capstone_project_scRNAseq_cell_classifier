"""
data_validation.py
==================
Validation module for the Cell Type Prediction Platform.

Primary entry point for the pipeline:
    validate_adata(adata)  — validates an already-loaded AnnData object

Standalone / CLI entry point:
    validate_file(filepath)  — loads the file via data_loader, then validates

Checks performed:
    - Matrix shape and dimensionality
    - Minimum cell and gene counts
    - NaN, Inf, and negative values (sparse-aware)
    - Duplicate cell barcodes and gene IDs
    - Matrix density (warns if unexpectedly dense)

Usage:
    from data_validation import validate_adata, ValidationResult

    result = validate_adata(adata)
    if result.is_valid:
        print(f"{result.n_cells} cells x {result.n_genes} genes — OK")
    else:
        print(result.errors)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Returned by validate_adata() and validate_file()."""

    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_cells: int = 0
    n_genes: int = 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def summary(self) -> str:
        lines = []
        if self.is_valid:
            lines.append(
                f"Validation passed  |  {self.n_cells:,} cells x {self.n_genes:,} genes"
            )
        else:
            lines.append(f"Validation failed  ({len(self.errors)} error(s))")
        for e in self.errors:
            lines.append(f"   ERROR   : {e}")
        for w in self.warnings:
            lines.append(f"   WARNING : {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

MIN_CELLS = 10
MIN_GENES = 50
HIGH_DENSITY_THRESHOLD = 0.8   # fraction of non-zero entries


# ---------------------------------------------------------------------------
# Primary entry point (pipeline use)
# ---------------------------------------------------------------------------

def validate_adata(adata) -> ValidationResult:
    """
    Validate an already-loaded AnnData object.

    This is the function called by the pipeline (Step 2).
    It does not reload the file — pass the AnnData returned by data_loader.

    Parameters
    ----------
    adata : anndata.AnnData
        Dataset to validate.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult(is_valid=True)

    if adata.X is None:
        result.add_error("AnnData has no expression matrix (adata.X is None).")
        return result

    if adata.n_obs == 0 or adata.n_vars == 0:
        result.add_error(
            f"Expression matrix is empty ({adata.n_obs} cells x {adata.n_vars} genes)."
        )
        return result

    if adata.n_obs < MIN_CELLS:
        result.add_error(
            f"Dataset contains only {adata.n_obs} cell(s); minimum required is {MIN_CELLS}."
        )

    if adata.n_vars < MIN_GENES:
        result.add_error(
            f"Dataset contains only {adata.n_vars} gene(s); minimum required is {MIN_GENES}."
        )

    if not result.is_valid:
        return result

    _check_numeric(adata.X, result)
    _check_duplicates(list(adata.obs_names), "cell barcode", result)
    _check_duplicates(list(adata.var_names), "gene ID", result)
    _check_density(adata.X, result)

    if result.is_valid:
        result.n_cells = adata.n_obs
        result.n_genes = adata.n_vars

    return result


# ---------------------------------------------------------------------------
# Secondary entry point (standalone / CLI use)
# ---------------------------------------------------------------------------

def validate_file(filepath: str | Path) -> ValidationResult:
    """
    Load a file via data_loader and validate it.

    This is a convenience wrapper for CLI and testing.
    The pipeline should call validate_adata() directly to avoid loading twice.
    """
    from data_loader import load_file

    path = Path(filepath)
    result = ValidationResult(is_valid=True)

    if not path.exists():
        result.add_error(f"File not found: '{path}'")
        return result

    if path.stat().st_size == 0:
        result.add_error("File is empty (0 bytes).")
        return result

    load_res = load_file(path)
    if not load_res.success:
        result.add_error(f"Could not load file: {load_res.error}")
        return result

    return validate_adata(load_res.adata)


# ---------------------------------------------------------------------------
# Internal checks
# ---------------------------------------------------------------------------

def _check_numeric(X, result: ValidationResult) -> None:
    """Check for NaN, Inf, and negative values. Operates on sparse data directly."""
    import scipy.sparse as sp

    data = X.data if sp.issparse(X) else np.asarray(X).ravel()

    nan_count = int(np.sum(np.isnan(data)))
    if nan_count > 0:
        result.add_error(
            f"Expression matrix contains {nan_count:,} NaN value(s). "
            f"Replace NaN with 0 or remove affected cells/genes before uploading."
        )

    inf_count = int(np.sum(np.isinf(data)))
    if inf_count > 0:
        result.add_error(
            f"Expression matrix contains {inf_count:,} infinite value(s)."
        )

    neg_count = int(np.sum(data < 0))
    if neg_count > 0:
        result.add_error(
            f"Expression matrix contains {neg_count:,} negative value(s). "
            f"Raw UMI counts must be non-negative."
        )

    if result.is_valid and data.dtype.kind == "f":
        n_non_integer = int(np.sum(data != np.floor(data)))
        if n_non_integer / max(data.size, 1) > 0.01:
            result.add_warning(
                f"{n_non_integer / data.size * 100:.1f}% of expression values are non-integer. "
                f"The platform expects raw UMI count matrices; normalised input may affect results."
            )


def _check_duplicates(ids: list, label: str, result: ValidationResult) -> None:
    """Detect duplicated cell barcodes or gene IDs."""
    seen: dict = {}
    for idx, item in enumerate(ids):
        seen.setdefault(item, []).append(idx)

    duplicates = {k: v for k, v in seen.items() if len(v) > 1}
    if duplicates:
        examples = list(duplicates.items())[:5]
        example_str = "; ".join(f"'{k}' at positions {v}" for k, v in examples)
        result.add_error(
            f"Found {len(duplicates)} duplicate {label}(s). "
            f"All identifiers must be unique. Examples: {example_str}"
            + ("  ..." if len(duplicates) > 5 else "")
        )


def _check_density(X, result: ValidationResult) -> None:
    """Warn if the matrix is unexpectedly dense (may indicate normalised data)."""
    import scipy.sparse as sp

    if sp.issparse(X):
        total = X.shape[0] * X.shape[1]
        non_zero = X.nnz
    else:
        total = X.size
        non_zero = int(np.count_nonzero(X))

    if total == 0:
        return

    density = non_zero / total
    if density > HIGH_DENSITY_THRESHOLD:
        result.add_warning(
            f"Matrix density is {density * 100:.1f}% non-zero. "
            f"Raw scRNA-seq count matrices are typically sparse (<50% non-zero). "
            f"This may indicate already-normalised data."
        )


# ---------------------------------------------------------------------------
# CLI:  python data_validation.py <file>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_validation.py <path_to_file>")
        sys.exit(1)

    res = validate_file(sys.argv[1])
    print(res.summary())
    sys.exit(0 if res.is_valid else 1)
