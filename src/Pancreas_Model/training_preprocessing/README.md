# Pancreas Dataset Preparation

This folder contains scripts for checking and preparing the **Tabula Sapiens - Pancreas** dataset for pancreas-specific cell type prediction.

## Dataset

The dataset was downloaded from **CZ CELLxGENE Discover** as an `.h5ad` file:

```text
Tabula_Sapiens_Pancreas.h5ad
```

It contains human pancreas single-cell RNA-seq data with cell metadata, gene metadata, cell type annotations, and multiple expression matrices.

## Scripts

### `check_pancreas_dataset_structure.py`

This script checks the original downloaded dataset before preprocessing.

It prints and saves:

- number of cells and genes
- available `obs`, `var`, `layers`, `obsm`, and `uns` fields
- possible label columns
- cell type distributions
- gene identifier information
- matrix statistics
- most count-like expression source

Main output folder:

```text
pancreas_structure_check/
```

The structure check showed that the best count-like matrix is:

```text
adata.layers["decontXcounts"]
```

and the most useful detailed label column is:

```text
adata.obs["cell_type"]
```

### `preprocess_train_pancreas.py`

This script converts the dataset into the same general structure used by the existing PBMC training pipeline.

Main changes:

- copies `adata.layers["decontXcounts"]` into `adata.X`
- copies `adata.var["feature_name"]` into `adata.var["gene_symbol"]`
- creates `adata.obs["barcode"]`
- creates `cell_type_original`
- creates `cell_type_standardized`
- creates `cell_type_level_1` from `compartment`
- creates `cell_type_level_2` from `broad_cell_class`
- creates `cell_type_level_3` from `cell_type`
- removes Level 3 cell types with fewer than 20 cells
- saves label distributions and preprocessing summary files

Output file:

```text
Tabula_Sapiens_Pancreas_preprocessed_for_training.h5ad
```

Output folder:

```text
pancreas_preprocessing_outputs/
```

## Label Hierarchy

```text
cell_type_level_1 = compartment
cell_type_level_2 = broad_cell_class
cell_type_level_3 = cell_type
```

Very rare Level 3 labels are removed before training to avoid unstable model training and unreliable evaluation results.

## How to Run

```bash
python check_pancreas_dataset_structure.py
python preprocess_train_pancreas.py
```

After preprocessing, use this file for model training:

```text
Tabula_Sapiens_Pancreas_preprocessed_for_training.h5ad
```
