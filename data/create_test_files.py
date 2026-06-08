"""
create_test_files.py
====================
Generates test files for GUI testing.

Run:
    python create_test_files.py

Outputs (all in test_files/ folder):

  ✅ VALID — should pass and produce predictions
    pbmc_valid_small.h5ad       150 cells  ·  2000 model genes  (small, fast)
    pbmc_valid_medium.h5ad      800 cells  ·  2000 model genes  (medium size)
    pbmc_valid_large.h5ad      2500 cells  ·  2000 model genes  (stress test)
    pbmc_valid_extra_genes.h5ad  300 cells  ·  5000 genes (2000 model + 3000 extra)
    expression_matrix.csv        100 cells  ·  2000 model genes  (CSV format)

  ⚠️  WARNING — valid but triggers warnings
    warn_dense.h5ad              200 cells  ·  2000 genes  (>80% non-zero → density warning)
    warn_float_counts.h5ad       150 cells  ·  2000 genes  (non-integer values → normalised warning)

  ❌ INVALID — should fail validation and show error in GUI
    bad_negative_values.h5ad     100 cells  ·  500 genes   (5% negative values)
    bad_too_few_cells.h5ad         5 cells  ·  300 genes   (< 10 cells)
    bad_too_few_genes.h5ad       100 cells  ·   20 genes   (< 50 genes)
    bad_nan_values.h5ad          100 cells  ·  500 genes   (NaN values)
    bad_duplicate_barcodes.h5ad  100 cells  ·  300 genes   (duplicate cell barcodes)
    bad_empty_matrix.h5ad          0 cells  ·    0 genes   (empty matrix)
"""

import sys
from pathlib import Path

try:
    import anndata as ad
    import joblib
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
except ImportError as e:
    print(f"Missing package: {e}")
    print("Run:  pip install anndata scipy numpy pandas joblib")
    sys.exit(1)

OUT = Path(__file__).parent / "test_files"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Load real gene names from the model bundle
# ---------------------------------------------------------------------------

BUNDLE_PATH = Path(__file__).parent / "LR_level1_no_weight_final_model_bundle.joblib"

print("Loading model bundle for gene list...")
try:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(BUNDLE_PATH)
    MODEL_GENES = list(bundle["training_gene_order"])   # 2000 ENSG IDs
    print(f"  Loaded {len(MODEL_GENES)} model genes from bundle.\n")
except Exception as e:
    print(f"  ERROR: Could not load bundle: {e}")
    print("  Falling back to fake ENSG gene names.")
    MODEL_GENES = [f"ENSG{10000 + i:08d}" for i in range(2000)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_count_matrix(n_cells: int, gene_list: list, lam: float = 3.0, seed: int = 0):
    """Realistic sparse UMI count matrix using the given gene list."""
    rng = np.random.default_rng(seed)
    dense = rng.poisson(lam=lam, size=(n_cells, len(gene_list))).astype(np.float32)
    return sp.csr_matrix(dense)


def make_adata(n_cells: int, gene_list: list, seed: int = 0) -> ad.AnnData:
    X = make_count_matrix(n_cells, gene_list, seed=seed)
    obs_names = [f"AAACCTGA{i:06d}-1" for i in range(n_cells)]
    return ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=gene_list),
    )


def save(adata: ad.AnnData, name: str, tag: str = "") -> None:
    path = OUT / name
    if sp.issparse(adata.X):
        nnz = adata.X.nnz
        total = adata.n_obs * adata.n_vars
        density = nnz / total * 100 if total > 0 else 0
    else:
        total = adata.X.size
        density = np.count_nonzero(adata.X) / total * 100 if total > 0 else 0
    adata.write_h5ad(path)
    tag_str = f"  [{tag}]" if tag else ""
    print(f"  {name:<40} {adata.n_obs:>5} cells x {adata.n_vars:>5} genes  "
          f"(density {density:.1f}%){tag_str}")


# ---------------------------------------------------------------------------
# ✅ VALID test files
# ---------------------------------------------------------------------------

print("=" * 65)
print("✅  VALID files (should pass and produce predictions)")
print("=" * 65)

# Small PBMC — fast for quick GUI tests
save(make_adata(150, MODEL_GENES, seed=1), "pbmc_valid_small.h5ad")

# Medium PBMC — realistic workload
save(make_adata(800, MODEL_GENES, seed=2), "pbmc_valid_medium.h5ad")

# Large PBMC — stress test
save(make_adata(2500, MODEL_GENES, seed=3), "pbmc_valid_large.h5ad")

# Extra genes (model genes + 3000 random extra) — tests gene alignment
extra_genes = [f"ENSG99{i:06d}" for i in range(3000)]
mixed_genes = MODEL_GENES + extra_genes          # model genes first, then extras
import random; random.seed(7); random.shuffle(mixed_genes)   # shuffle order
save(make_adata(300, mixed_genes, seed=4), "pbmc_valid_extra_genes.h5ad",
     tag="has 3000 extra genes → alignment trims them")

# CSV format — cells as rows, genes as columns
print()
print("  expression_matrix.csv              ", end="")
n_cells_csv = 100
rng = np.random.default_rng(5)
matrix_csv = rng.poisson(lam=3, size=(n_cells_csv, len(MODEL_GENES))).astype(float)
cell_ids = [f"cell_{i:04d}" for i in range(n_cells_csv)]
df_csv = pd.DataFrame(matrix_csv, index=cell_ids, columns=MODEL_GENES)
df_csv.to_csv(OUT / "expression_matrix.csv")
print(f"  {n_cells_csv:>5} cells x {len(MODEL_GENES):>5} genes  (CSV format)")


# ---------------------------------------------------------------------------
# ⚠️  WARNING files (valid but will show pipeline warnings)
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("⚠️   WARNING files (pass validation but trigger warnings)")
print("=" * 65)

# Dense matrix (>80% non-zero) → "may be normalised data" warning
rng_dense = np.random.default_rng(20)
dense_X = (rng_dense.random((200, len(MODEL_GENES))) * 10 + 0.1).astype(np.float32)
obs_dense = [f"DENSE{i:05d}" for i in range(200)]
adata_dense = ad.AnnData(
    X=dense_X,
    obs=pd.DataFrame(index=obs_dense),
    var=pd.DataFrame(index=MODEL_GENES),
)
save(adata_dense, "warn_dense.h5ad", tag="dense → density warning expected")

# Float (non-integer) counts → "normalised input" warning
adata_float = make_adata(150, MODEL_GENES, seed=21)
adata_float.X = adata_float.X.astype(np.float32)
# add small random decimal parts
rng_f = np.random.default_rng(22)
float_dense = adata_float.X.toarray()
float_dense += rng_f.random(float_dense.shape).astype(np.float32) * 0.5
adata_float.X = sp.csr_matrix(float_dense)
save(adata_float, "warn_float_counts.h5ad", tag="non-integer → normalised warning")


# ---------------------------------------------------------------------------
# ❌ INVALID files (should fail and show clear error messages in GUI)
# ---------------------------------------------------------------------------

print()
print("=" * 65)
print("❌  INVALID files (should fail validation — test error display)")
print("=" * 65)

# Negative values
adata_neg = make_adata(100, [f"ENSG{10000 + i:08d}" for i in range(500)], seed=10)
neg_mask = np.random.default_rng(99).random(adata_neg.X.data.shape) < 0.05
adata_neg.X.data[neg_mask] *= -1
save(adata_neg, "bad_negative_values.h5ad", tag="~5% negative → ERROR")

# Too few cells (< MIN_CELLS = 10)
save(make_adata(5, [f"ENSG{10000 + i:08d}" for i in range(300)], seed=11),
     "bad_too_few_cells.h5ad", tag="5 cells < 10 minimum → ERROR")

# Too few genes (< MIN_GENES = 50)
save(make_adata(100, [f"ENSG{10000 + i:08d}" for i in range(20)], seed=12),
     "bad_too_few_genes.h5ad", tag="20 genes < 50 minimum → ERROR")

# NaN values
adata_nan = make_adata(100, [f"ENSG{10000 + i:08d}" for i in range(500)], seed=13)
nan_dense = adata_nan.X.toarray()
nan_mask = np.random.default_rng(88).random(nan_dense.shape) < 0.03
nan_dense[nan_mask] = np.nan
adata_nan.X = sp.csr_matrix(nan_dense)
save(adata_nan, "bad_nan_values.h5ad", tag="NaN values → ERROR")

# Duplicate barcodes
adata_dup = make_adata(100, [f"ENSG{10000 + i:08d}" for i in range(300)], seed=14)
dup_names = list(adata_dup.obs_names)
dup_names[5] = dup_names[0]    # duplicate barcode at position 5
dup_names[20] = dup_names[10]  # duplicate barcode at position 20
adata_dup.obs_names = dup_names
save(adata_dup, "bad_duplicate_barcodes.h5ad", tag="2 duplicate barcodes → ERROR")

# Empty matrix (0 × 0)
try:
    adata_empty = ad.AnnData(
        X=sp.csr_matrix((0, 0)),
        obs=pd.DataFrame(index=[]),
        var=pd.DataFrame(index=[]),
    )
    save(adata_empty, "bad_empty_matrix.h5ad", tag="0 cells × 0 genes → ERROR")
except Exception as exc:
    print(f"  bad_empty_matrix.h5ad                skipped ({exc})")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

all_files = sorted(OUT.glob("*"))
print()
print("=" * 65)
print(f"Done. {len(all_files)} file(s) saved to: {OUT}/")
print()
print("How to use:")
print("  1. streamlit run gui.py")
print("  2. Upload a file from test_files/")
print("  3. Select tissue = PBMC for all pbmc_* files")
print()
print("Expected behaviour:")
print("  pbmc_valid_*          → Full pipeline succeeds, predictions shown")
print("  expression_matrix.csv → Full pipeline succeeds (CSV input)")
print("  warn_*                → Pipeline succeeds but warnings appear")
print("  bad_*                 → Validation ERROR shown, pipeline stops")
