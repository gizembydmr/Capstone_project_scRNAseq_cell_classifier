# Model Training Datasets and GUI Test Subsets

The full datasets are too large to store on GitHub. They are stored in Google Drive.

Download them from:
https://drive.google.com/drive/folders/1rnsStUR6nyYy8WwGVRJ9HPcPG9p4kGGD?usp=drive_link

The Google Drive folder contains the large `.h5ad` datasets used for model training and GUI/backend testing.

### PBMC68k dataset files

* `pbmc68k_annotated_with_levels.h5ad` contains the raw annotated PBMC68k dataset with hierarchical cell-type labels.
* `pbmc68k_preprocessed_for_training.h5ad` is the preprocessed training dataset used for machine learning model development.
* `pbmc68k_hvg_list.csv` stores the selected highly variable genes used during training.

### Pancreas dataset files

* `Tabula_Sapiens_Pancreas_original.h5ad` is the original downloaded pancreas dataset.
* `Tabula_Sapiens_Pancreas_annotated_with_levels.h5ad` contains the raw/count-like pancreas dataset with added hierarchical labels.
* `Tabula_Sapiens_Pancreas_preprocessed_for_training.h5ad` is the preprocessed training dataset used for pancreas model development.
* `pancreas_training_preprocessing_metadata.json` stores preprocessing information such as filtering parameters, normalization target, and HVG settings.

## GUI Test Subsets

GUI test subsets are used to simulate user-uploaded datasets during backend and interface testing. These files are intentionally created from raw/count-like data instead of preprocessed model-training data.

They are used to test the full inference workflow:

```text
data loading
data validation
inference preprocessing
gene alignment
model prediction
GUI result display
result export
```

These files are for integration testing only. They do not replace the held-out test results used for final model evaluation.

## PBMC68k GUI Test Subset

pbmc68k_gui_test_2401cells.h5ad is a small semi-balanced PBMC68k test dataset for MVP backend and GUI integration.

It was created from: pbmc68k_annotated_with_levels.h5ad

The source file contains raw UMI counts in `adata.X` and hierarchical cell-type labels.

Before saving the GUI input file, the true label columns were removed from `adata.obs` to simulate a real user-uploaded dataset.

### Source split

The subset was sampled from the untouched held-out test split used during PBMC model development.

### Sampling strategy

The PBMC68k dataset is highly imbalanced, especially because T cells are very abundant. To avoid creating a GUI test file dominated by one class, a semi-balanced sampling strategy was used across Level 1 labels.

Final Level 1 distribution:

| Cell type  | Number of cells |
| ---------- | --------------: |
| B cell     |             500 |
| Monocyte   |             500 |
| NK cell    |             500 |
| T cell     |             500 |
| Dendritic  |             355 |
| Progenitor |              46 |

Total number of cells:

```text
2401
```

## Pancreas GUI Test Subset

pancreas_gui_test_all_test_cells.h5ad is a raw/count-like pancreas test dataset for backend and GUI integration.

It was created from: Tabula_Sapiens_Pancreas_annotated_with_levels.h5ad

This source file contains the full-gene pancreas dataset with hierarchical labels.

Before saving the GUI input file, the true label columns were removed from `adata.obs` to simulate a real user-uploaded dataset.

### Source split

The subset was created from the untouched 20% held-out test split used during pancreas Level 3 (finest cell labels) model comparison.

The saved test indices were first mapped back to cell IDs using the preprocessed pancreas training dataset, then the same cells were selected from the raw/full-gene annotated pancreas dataset.

### Sampling strategy

For pancreas, all held-out test cells were used instead of downsampling. The test set is small enough for GUI/backend testing and keeping all cells preserves rare Level 3 classes.

Final dataset size:

```text
2818 cells × 60606 genes
```

Final Level 3 distribution:

| Cell type                       | Number of cells |
| ------------------------------- | --------------: |
| pancreatic acinar cell          |            1096 |
| pancreatic ductal cell          |             544 |
| endothelial cell                |             465 |
| macrophage                      |             318 |
| CD8-positive, alpha-beta T cell |             124 |
| pancreatic stellate cell        |              88 |
| classical monocyte              |              78 |
| fibroblast                      |              39 |
| type B pancreatic cell          |              20 |
| B cell                          |              11 |
| natural killer cell             |              10 |
| pancreatic A cell               |              10 |
| intermediate monocyte           |               8 |
| CD4-positive, alpha-beta T cell |               7 |

Total number of cells:

```text
2818
```
