# PBMC68k Preprocessing Pipeline

This folder contains scripts used to preprocess and prepare the PBMC68k dataset for downstream analysis in Python.

---

## Dataset source

We use the PBMC68k dataset from:
https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis

Based on:
Zheng et al., 2017 – *Massively parallel digital transcriptional profiling of single cells*
https://www.nature.com/articles/ncomms14049 

---

## Pipeline Overview

### 1. R preprocessing and label generation

Script: `Modified_main_process_68k_pbmc.R`

**Input:**
- `pbmc68k_data.rds`
- `all_pure_select_11types.rds`

**What it does:**
- Normalizes gene expression (UMI-based)
- Selects variable genes
- Performs PCA and t-SNE
- Assigns cell type labels by correlation with purified PBMC reference profiles

**Outputs:**
- `pbmc_matrix.mtx` → sparse gene expression matrix (cells × genes)
- `pbmc_labels.csv` → cell barcodes + assigned cell type labels
- `pbmc_gene_metadata.csv` → Ensembl IDs and gene symbols
- `pbmc_barcodes.csv` → list of cell barcodes
- `pbmc_dataset_metadata.csv` → dataset-level metadata

These outputs are used to transfer the dataset from R to Python.

---

### 2. Build AnnData object

Script: `create_h5ad.py`

**Input:**
- `pbmc_matrix.mtx`
- `pbmc_gene_metadata.csv`
- `pbmc_labels.csv`
- `pbmc_dataset_metadata.csv`

**Output:**
- `pbmc68k_annotated.h5ad`

**What it does:**
- Constructs a Scanpy AnnData object
- Stores gene expression, cell metadata, gene metadata, and dataset metadata in a unified format

---

### 3. Dataset validation

Script: `check_dataset.py`

**Input:**
- `pbmc68k_annotated.h5ad`

**What it checks:**
- matrix dimensions (cells vs genes)
- missing values (labels, barcodes, gene metadata)
- uniqueness of barcodes and gene IDs
- sparsity of expression matrix
- cell type distribution
- basic count statistics

Ensures the dataset is consistent and ready for analysis.

---

### 4. Add hierarchical cell type labels

Script: `add_label_hierarchy.py`

**Input:**
- `pbmc68k_annotated.h5ad`

**Output:**
- `pbmc68k_annotated_with_levels.h5ad`

**What it does:**
- Adds hierarchical label levels:
  - `cell_type_standardized`
  - `cell_type_level_1`
  - `cell_type_level_2`
  - `cell_type_level_3`

**Example:**
CD8+ Cytotoxic T
→ Level 1: T cell
→ Level 2: CD8 T
→ Level 3: CD8 cytotoxic T

**Why:**

This enables modeling at different biological resolutions (e.g., general cell types vs detailed subtypes).

## Final Output

The final processed dataset: `pbmc68k_annotated_with_levels.h5ad`
