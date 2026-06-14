# Pancreas Dataset Preparation

This folder contains scripts for checking, restructuring, and preprocessing the **Tabula Sapiens - Pancreas** dataset for pancreas-specific cell type prediction.

## Dataset

The original dataset was downloaded from **CZ CELLxGENE Discover** as an `.h5ad` file.

Original input file:

```text
Tabula_Sapiens_Pancreas_original.h5ad
```

The dataset contains human pancreas single-cell RNA-seq data with cell metadata, gene metadata, cell type annotations, and multiple expression matrices.

## Pipeline Overview

```text
Tabula_Sapiens_Pancreas_original.h5ad
        ↓
Tabula_Sapiens_Pancreas_annotated_with_levels.h5ad
        ↓
outputs/Tabula_Sapiens_Pancreas_preprocessed_for_training.h5ad
```

## Scripts

### `01_check_pancreas_dataset_structure.py`

This script checks the original downloaded dataset before any modification.

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

This script does not create a new dataset.

### `02_update_data_structure_pancreas.py`

This script converts the original CELLxGENE dataset into the same general annotated structure used in the PBMC68k pipeline.

Input:

```text
Tabula_Sapiens_Pancreas_original.h5ad
```

Output:

```text
Tabula_Sapiens_Pancreas_annotated_with_levels.h5ad
```

Main changes:

- creates a clean AnnData object using `adata.layers["decontXcounts"]` as `adata.X`
- copies `adata.var["feature_name"]` into `adata.var["gene_symbol"]`
- creates `adata.obs["barcode"]`
- creates `cell_type_original`
- creates `cell_type_standardized`
- creates `cell_type_level_1` from `compartment`
- creates `cell_type_level_2` from `broad_cell_class`
- creates `cell_type_level_3` from `cell_type`
- removes Level 3 cell types with fewer than 20 cells
- saves label distributions and structure summary files

Output folder:

```text
pancreas_data_structure_outputs/
```

This output dataset is the pancreas equivalent of:

```text
pbmc68k_annotated_with_levels.h5ad
```

### `03_preprocess_train_pancreas.py`

This script performs the training preprocessing step, similar to the PBMC68k `preprocess_train.py` script.

Input:

```text
Tabula_Sapiens_Pancreas_annotated_with_levels.h5ad
```

Outputs:

```text
outputs/Tabula_Sapiens_Pancreas_preprocessed_for_training.h5ad
outputs/pancreas_hvg_list.csv
outputs/pancreas_training_preprocessing_metadata.json
```

Main steps:

- computes quality control metrics
- filters low-quality cells
- removes genes with no expression
- stores raw counts in `adata.layers["counts"]`
- normalizes counts using median library-size scaling
- applies `log1p` transformation
- selects 2000 highly variable genes
- stores the full normalized/log-transformed gene space in `adata.raw`
- saves the final HVG list and exact gene order for inference-time gene alignment

This output dataset is the pancreas equivalent of:

```text
pbmc68k_preprocessed_for_training.h5ad
```

## Label Hierarchy

The prepared pancreas dataset uses the following label hierarchy:

```text
cell_type_level_1 = compartment
cell_type_level_2 = broad_cell_class
cell_type_level_3 = cell_type
```

Very rare Level 3 labels are removed before training if they contain fewer than 20 cells. This is done to avoid unstable model training and unreliable evaluation results for classes with too few examples.

## Final Training Dataset

For model training and model comparison:

```text
outputs/Tabula_Sapiens_Pancreas_preprocessed_for_training.h5ad
```

For inference-time gene alignment:

```text
outputs/pancreas_hvg_list.csv
```
