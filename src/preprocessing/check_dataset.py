import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from pathlib import Path

# path of the .h5ad file
DATA_PATH = Path(r"C:\Users\ferid\Downloads\capstone_demo\pbmc68k_annotated.h5ad")

# load AnnData object
adata = sc.read_h5ad(DATA_PATH)

# basic dataset summary
print("=== BASIC DATASET SUMMARY ===")
print(adata)
print()

# number of cells and genes
print("=== NUMBER OF CELLS AND GENES ===")
print("Number of cells:", adata.n_obs)
print("Number of genes:", adata.n_vars)
print()

# matrix shape check
print("=== MATRIX SHAPE CHECK ===")
print("Expression matrix shape:", adata.X.shape)
print("Number of rows in obs (cells metadata):", len(adata.obs))
print("Number of rows in var (gene metadata):", len(adata.var))
print()

# first rows of cell metadata
print("=== FIRST 5 CELL METADATA ROWS ===")
print(adata.obs.head())
print()

# first rows of gene metadata
print("=== FIRST 5 GENE METADATA ROWS ===")
print(adata.var.head())
print()

# missing label check
print("=== MISSING LABEL CHECK ===")
missing_labels = adata.obs["cell_type"].isna().sum()
print("Missing cell_type labels:", missing_labels)
print()

# missing barcode check
print("=== MISSING BARCODE CHECK ===")
missing_barcodes = adata.obs["barcode"].isna().sum()
print("Missing barcode values:", missing_barcodes)
print()

# missing gene metadata check
print("=== MISSING GENE METADATA CHECK ===")
missing_gene_symbols = adata.var["gene_symbol"].isna().sum()
missing_ensembl_ids = adata.var_names.isna().sum() if hasattr(adata.var_names, "isna") else 0
print("Missing gene_symbol values:", missing_gene_symbols)
print("Missing Ensembl IDs in var_names:", missing_ensembl_ids)
print()

# cell type distribution
print("=== CELL TYPE DISTRIBUTION ===")
cell_type_counts = adata.obs["cell_type"].value_counts()
print(cell_type_counts)
print()

# sparsity check
print("=== SPARSITY CHECK ===")
print("Is adata.X sparse?", sparse.issparse(adata.X))
print()

# small matrix preview
print("=== SMALL EXPRESSION MATRIX PREVIEW (5x5) ===")
print(adata.X[:5, :5])
print()

# basic count statistics
print("=== BASIC COUNT STATISTICS ===")

cell_sums = np.asarray(adata.X.sum(axis=1)).flatten()
gene_sums = np.asarray(adata.X.sum(axis=0)).flatten()

print("Mean total counts per cell:", cell_sums.mean())
print("Minimum total counts in a cell:", cell_sums.min())
print("Maximum total counts in a cell:", cell_sums.max())
print()

print("Mean total counts per gene:", gene_sums.mean())
print("Minimum total counts in a gene:", gene_sums.min())
print("Maximum total counts in a gene:", gene_sums.max())
print()

# barcode uniqueness check
print("=== BARCODE UNIQUENESS CHECK ===")
print("Unique barcodes:", adata.obs["barcode"].nunique())
print("Total barcodes:", len(adata.obs["barcode"]))
print("Are barcodes unique?", adata.obs["barcode"].nunique() == adata.n_obs)
print()

# gene symbol uniqueness check
print("=== GENE SYMBOL UNIQUENESS CHECK ===")
print("Unique gene symbols:", adata.var["gene_symbol"].nunique())
print("Total genes:", len(adata.var["gene_symbol"]))
print("Are gene symbols unique?", adata.var["gene_symbol"].nunique() == adata.n_vars)
print()

# Ensembl ID uniqueness check
print("=== ENSEMBL ID UNIQUENESS CHECK ===")
print("Unique Ensembl IDs:", adata.var_names.nunique())
print("Total var_names:", len(adata.var_names))
print("Are Ensembl IDs unique?", adata.var_names.nunique() == adata.n_vars)
print()

# preview of var_names
print("=== FIRST 5 var_names (should be Ensembl IDs) ===")
print(list(adata.var_names[:5]))
print()

# dataset metadata check
print("=== DATASET METADATA CHECK ===")
print(adata.uns["dataset_metadata"])

# plot cell type distribution
cell_type_counts.plot(kind="bar")
plt.title("Cell Type Distribution")
plt.xlabel("Cell Type")
plt.ylabel("Number of Cells")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


