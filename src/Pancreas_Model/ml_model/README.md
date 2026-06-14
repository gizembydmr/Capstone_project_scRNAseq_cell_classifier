# Pancreas Model Development

This folder contains scripts for comparing machine learning models and training the final selected model for **Tabula Sapiens pancreas Level 3 cell-type prediction**.

The input dataset is:

```text
Tabula_Sapiens_Pancreas_preprocessed_for_training.h5ad
```

The prediction label is:

```text
cell_type_level_3
```

## Scripts

### `01_compare_all_models_level3_pancreas.py`

This script compares three machine learning model families with and without class weighting:

* Logistic Regression without class weighting
* Logistic Regression with balanced class weighting
* Linear SVM without class weighting
* Linear SVM with balanced class weighting
* Random Forest without class weighting
* Random Forest with balanced class weighting

The script creates an **80/20 stratified train-test split**, performs **5-fold stratified cross-validation** on the 80% development set, and evaluates each model once on the untouched 20% test set.

Main outputs:

```text
pancreas_all_models_level3_CV_results.xlsx
pancreas_all_models_level3_metadata.json
pancreas_all_models_level3_split_indices.npz
figures/
```

The Excel file contains the cross-validation summary, final test metrics, class-level reports, label distributions, and confusion matrices.

## Final Model Selection
<img width="3870" height="1106" alt="pancreas_all_models_level3_presentation_summary_table" src="https://github.com/user-attachments/assets/ca7da243-ed40-488e-8430-0dfd1a6dd1eb" />

The final model was selected based mainly on **Macro F1** and **balanced accuracy**, because the Level 3 pancreas labels are imbalanced. In this setting, accuracy alone can hide poor performance on rare cell types.

Although Random Forest achieved the highest cross-validation accuracy, Logistic Regression with balanced class weighting achieved the best Macro F1 and balanced accuracy:

```text
LR_balanced CV Macro F1: 0.853
LR_balanced CV balanced accuracy: 0.855

RF_no_weight CV Macro F1: 0.820
RF_no_weight CV balanced accuracy: 0.817
```

<img width="5215" height="4168" alt="CV_LR_balanced_confusion_heatmap_normalized" src="https://github.com/user-attachments/assets/d8c0fc8c-bcdf-459d-a77a-ff7084e4b4ca" />

<img width="5215" height="4168" alt="CV_RF_no_weight_confusion_heatmap_normalized" src="https://github.com/user-attachments/assets/f9cd6eb9-8edf-446e-aa4a-2f9c3703b2b8" />

This means Random Forest performed well on abundant classes, but it was less reliable for rare Level 3 cell types. The normalized confusion matrix heatmaps also show that the LR balanced model gives a better overall balance across classes.

Therefore, the final selected model is:

```text
LR_balanced
```

This model provides the best trade-off between overall accuracy and rare-class performance for pancreas Level 3 cell-type prediction.

### `02_train_final_LR_balanced_model_level3.py`

This script trains the final selected pancreas model:

```text
Logistic Regression with class_weight="balanced"
```

It uses the same saved 80% development set from the model comparison script and evaluates the final model once on the same untouched 20% test set.

The script saves the trained model bundle for platform integration. The bundle includes the model, label encoder, class names, training gene order, gene symbols, and preprocessing settings required for inference.

Main outputs:

```text
pancreas_LR_balanced_level3_final_model_bundle.joblib
pancreas_LR_balanced_level3_final_model_metadata.json
pancreas_LR_balanced_level3_final_test_results.xlsx
figures/
```

