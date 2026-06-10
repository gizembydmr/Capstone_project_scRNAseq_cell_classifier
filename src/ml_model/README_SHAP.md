# SHAP Explainability for Final Logistic Regression Model

This README explains the SHAP explainability workflow added for the final Level 1 PBMC cell-type prediction model.

## Script

The SHAP workflow is implemented in:

```text
src/ml_model/06_explain_final_LR_with_SHAP.py
```

## Purpose

The goal of this script is to explain which genes contribute most strongly to the final model's cell-type predictions.

The final selected model is a Logistic Regression classifier. Since Logistic Regression is a linear classification model, the script uses:

```python
shap.LinearExplainer
```

This provides gene-level contribution scores for each predicted cell-type class.

## Input files

The script expects the final trained model bundle at:

```text
src/ml_model/models/LR_level1_no_weight_final_model_bundle.joblib
```

This model bundle contains:

- trained Logistic Regression model
- label encoder
- class names
- training HVG gene order
- gene symbols
- preprocessing parameters, including `target_sum`
- metadata required for inference integration

The script currently uses the test/query dataset:

```text
data/pbmc68k_gui_test_2401cells.h5ad
```

## Workflow

The script performs the following steps:

1. Loads the final Logistic Regression model bundle.
2. Reads the query/test `.h5ad` dataset.
3. Applies inference preprocessing using the preprocessing parameters saved in the model package:
   - `target_sum`
   - `min_counts`
   - `min_genes`
4. Aligns query genes to the saved 2000-gene training HVG order.
5. Runs cell-type prediction.
6. Computes SHAP values using `shap.LinearExplainer`.
7. Saves global, class-specific, predicted-group, and cell-level SHAP outputs.

## Output folder

All outputs are saved under:

```text
outputs/shap_LR_level1/
```

## Main output files

### Prediction output

```text
predicted_cell_types.csv
```

Contains predicted cell-type labels for retained cells.

### Explained cell metadata

```text
shap_explained_cells.csv
```

Contains the sampled cells used for SHAP explanation and their predicted labels.

### Global SHAP importance

```text
shap_global_gene_importance.csv
```

Summarizes global gene importance across explained cells and all model classes.

Important column:

```text
mean_abs_shap
```

A larger `mean_abs_shap` value means the gene had a stronger overall contribution to model predictions.

### Class-specific SHAP importance

```text
shap_class_specific_gene_importance.csv
```

Summarizes gene importance separately for each model class.

This output answers questions such as:

```text
Which genes are most important for the T cell class?
Which genes are most important for the B cell class?
```

### Predicted-group SHAP importance

```text
shap_gene_importance_by_predicted_group.csv
```

This is the main GUI/report-oriented group-level output.

For each predicted group, the script first selects cells predicted as that group. Then it summarizes SHAP values for the same class across only those cells.

Columns:

```text
predicted_group
rank
ensembl_id
gene_symbol
mean_abs_shap
mean_shap
n_cells
```

Column meanings:

- `predicted_group`: the cell-type group being summarized
- `rank`: gene importance rank within that predicted group
- `ensembl_id`: Ensembl gene ID
- `gene_symbol`: gene symbol
- `mean_abs_shap`: average contribution strength across cells in that predicted group
- `mean_shap`: average signed contribution
- `n_cells`: number of explained cells in that predicted group

`rank` restarts for each predicted group.

### Cell-level SHAP explanations

```text
shap_top_genes_per_cell.csv
```

This file provides top gene explanations for each explained cell.

Columns:

```text
cell_id
predicted_class
rank
ensembl_id
gene_symbol
shap_value
abs_shap_value
model_input_expression
```

Column meanings:

- `cell_id`: barcode or cell name
- `predicted_class`: model prediction for that cell
- `rank`: gene importance rank within that cell
- `ensembl_id`: Ensembl gene ID
- `gene_symbol`: gene symbol
- `shap_value`: signed contribution to the predicted class
- `abs_shap_value`: absolute contribution strength used for ranking
- `model_input_expression`: normalized/log-transformed expression value used by the model

For each explained cell, the top genes are selected using `abs_shap_value`.

## Plot outputs

The script also saves PNG plots for reporting and GUI use.

### Global plot

```text
shap_global_top_genes.png
```

Shows the top global SHAP genes ranked by `mean_abs_shap`.

### Class-specific plots

```text
shap_top_genes_<class>.png
```

Examples:

```text
shap_top_genes_T_cell.png
shap_top_genes_B_cell.png
shap_top_genes_NK_cell.png
```

These plots show top genes for each model class.

### Predicted-group GUI plots

```text
shap_top_genes_by_predicted_group_<group>.png
```

Examples:

```text
shap_top_genes_by_predicted_group_T_cell.png
shap_top_genes_by_predicted_group_B_cell.png
shap_top_genes_by_predicted_group_NK_cell.png
```

These are intended for GUI/report visualization.

For example, if the GUI dropdown selects `T cell`, the GUI can display:

```text
shap_top_genes_by_predicted_group_T_cell.png
```

and filter the table:

```text
shap_gene_importance_by_predicted_group.csv
```

where:

```text
predicted_group == "T cell"
```

The predicted-group plots show the top genes ranked by:

```text
mean_abs_shap
```

Suggested GUI labels:

```text
Title: Top genes contributing to <predicted_group> predictions
x-axis: Mean absolute SHAP value
y-axis: Gene
```

## Interpretation notes

`mean_abs_shap` measures contribution strength. It does not show direction, only magnitude.

`mean_shap` and `shap_value` are signed values:

- positive values generally support the class prediction
- negative values generally argue against the class prediction

For report interpretation, predicted-group outputs are usually more directly useful than global outputs because they summarize explanations only among cells predicted as a specific cell type.

## How to run

From the project root:

```bash
python src/ml_model/06_explain_final_LR_with_SHAP.py
```

If SHAP is not installed:

```bash
pip install shap
```

## Notes

The script is standalone and does not modify the GUI directly.

It generates CSV and PNG outputs that can later be used by the GUI or final report.
