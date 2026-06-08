import scanpy as sc
import pandas as pd
from scipy.io import mmread
from pathlib import Path

# paths
DATA_DIR = Path(r"C:\Users\ferid\Downloads\PBMC68k\results") 
OUTPUT_DIR = Path(r"C:\Users\ferid\Downloads\capstone_demo")

# load matrix
X = mmread(DATA_DIR / "pbmc_matrix.mtx").tocsr()

# load metadata
genes = pd.read_csv(DATA_DIR / "pbmc_gene_metadata.csv")
labels = pd.read_csv(DATA_DIR / "pbmc_labels.csv")
dataset_meta = pd.read_csv(DATA_DIR / "pbmc_dataset_metadata.csv")

# create AnnData
adata = sc.AnnData(X)

# gene metadata (use Ensembl IDs as index, gene symbols as column)
adata.var["gene_symbol"] = genes["gene_symbol"].values
adata.var_names = genes["ensembl_id"].values

# cell metadata
adata.obs["barcode"] = labels["barcode"].values
adata.obs["cell_type"] = labels["cell_type"].values
adata.obs_names = labels["barcode"].values

# dataset metadata
adata.uns["dataset_metadata"] = dict(zip(dataset_meta["field"], dataset_meta["value"]))

# save
adata.write(OUTPUT_DIR / "pbmc68k_annotated.h5ad")

print(adata)
