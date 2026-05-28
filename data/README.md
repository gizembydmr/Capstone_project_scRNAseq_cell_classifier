# Model Training Dataset

The dataset is too large to store on GitHub.

Download it from: 
https://drive.google.com/drive/folders/1rnsStUR6nyYy8WwGVRJ9HPcPG9p4kGGD?usp=drive_link 



# GUI Test Subset

A small semi-balanced PBMC68k test dataset for MVP backend and GUI integration.

## File

`pbmc68k_gui_test_2401cells.h5ad`

## Purpose

This dataset is intended to test the full inference workflow of the platform:

1. data loading
2. data validation
3. inference preprocessing
4. gene alignment
5. model prediction
6. GUI result display and export

## Source

The subset was created from:

`pbmc68k_annotated_with_levels.h5ad`

This source file contains raw UMI counts in `adata.X`, together with cell-type metadata and hierarchical labels.

## Why this file is not preprocessed

This file is intentionally kept in raw-count form so that the inference preprocessing pipeline can be tested.

It is not the preprocessed model-training dataset.

The final model expects uploaded/query data to be:

- filtered using the saved preprocessing parameters
- normalized using the training `target_sum` stored in the model package
- log1p transformed
- aligned to the saved training HVG gene order

## Sampling strategy

The subset was sampled from the held-out test split using a semi-balanced strategy across Level 1 cell-type labels.

The original PBMC68k dataset is highly imbalanced, especially with many T cells and few Progenitor cells. To avoid creating a GUI test file dominated by T cells, the number of cells per class was capped where possible.

Final Level 1 distribution:

| Cell type | Number of cells |
|---|---:|
| B cell | 500 |
| Monocyte | 500 |
| NK cell | 500 |
| T cell | 500 |
| Dendritic | 355 |
| Progenitor | 46 |

Total number of cells: 2401

