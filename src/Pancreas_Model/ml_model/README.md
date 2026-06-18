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

This script trains the selected pancreas model:

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

### `03_compare_unassigned_thresholds_pancreas.py`

This script evaluates confidence thresholds for adding an `Unassigned` label to the final pancreas Level 3 Logistic Regression model.

The script trains the selected balanced Logistic Regression model on the saved 80% development set and evaluates confidence thresholds on the untouched 20% test set. For each cell, the model calculates class probabilities using `predict_proba`. The highest class probability is used as the confidence score. If this score is below the tested threshold, the final prediction is changed to `Unassigned`.

The tested thresholds are:

```text
0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
```

The script uses the same saved split file from the pancreas model comparison script:

```text
pancreas_all_models_level3_split_indices.npz
```

Main outputs:

```text
pancreas_LR_balanced_level3_unassigned_threshold_comparison.xlsx
pancreas_LR_balanced_level3_unassigned_threshold_comparison_metadata.json
figures/
```

The Excel file contains:

* threshold summary table
* presentation summary table
* baseline metrics without `Unassigned`
* test predictions with confidence scores
* final labels for each tested threshold
* classification reports for each threshold
* raw and normalized confusion matrices for each threshold

The figures folder contains:

* presentation-ready threshold summary table
* assigned vs unassigned cell percentage plot
* confidence threshold trade-off plot
* normalized confusion matrix heatmaps for each threshold

<img width="3264" height="1692" alt="pancreas_LR_balanced_level3_unassigned_threshold_assigned_unassigned_plot" src="https://github.com/user-attachments/assets/bbc5e0f3-0f63-4c5a-bea8-f33d9e975de9" />

<img width="4470" height="2669" alt="pancreas_LR_balanced_level3_unassigned_threshold_summary_table" src="https://github.com/user-attachments/assets/3c025481-1e05-485b-a821-f918efab97be" />

<img width="3282" height="1957" alt="pancreas_LR_balanced_level3_unassigned_threshold_tradeoff_plot" src="https://github.com/user-attachments/assets/e6da6b30-6342-462b-aa66-70f4006481cc" />

The threshold comparison showed that the pancreas model produced very high prediction confidence overall. Lower thresholds such as 0.30 to 0.45 had no effect, keeping 100.00% of cells assigned. A threshold of 0.50 marked only 0.04% of cells as `Unassigned`, so it did not meaningfully activate the rejection mechanism.

<img width="4599" height="4012" alt="pancreas_LR_balanced_level3_unassigned_threshold_85_heatmap" src="https://github.com/user-attachments/assets/d9c08932-0f29-488a-b118-7541eeb9f038" />

The final pancreas threshold was selected using the same coverage-based rule used for the PBMC model: choose the highest tested threshold that preserves at least 95% assigned-cell coverage. Under this rule, threshold **0.85** was selected. At this threshold:

```text
Assigned cells: 95.71%
Unassigned cells: 4.29%
Accuracy on assigned cells: 98.15%
Macro F1 on assigned cells: 88.19%
Strict accuracy: 93.93%
```

The 0.80 threshold also kept high coverage, but it gave lower assigned-cell accuracy and Macro F1. The 0.90 threshold gave slightly higher assigned-cell scores, but assigned-cell coverage decreased to 94.96%, which fell below the 95% coverage requirement. Therefore, 0.85 was selected as the final confidence threshold for the pancreas model.

### `04_train_final_LR_balanced_model_level3_with_unassigned_threshold_085.py`

This script trains the final pancreas Level 3 balanced Logistic Regression model and saves it with the confidence-based `Unassigned` feature.

The model uses:

```text
Logistic Regression with class_weight="balanced"
```

and applies the selected confidence threshold:

```text
confidence threshold = 0.85
```

The model is trained on the full saved 80% development set and evaluated once on the untouched 20% test set. During evaluation, the model first predicts class probabilities for each cell. The highest probability is stored as the confidence score. If the confidence score is lower than 0.85, the final label is changed to `Unassigned`; otherwise, the predicted pancreas Level 3 cell-type label is kept.

Main outputs:

```text
pancreas_LR_balanced_level3_final_model_bundle_with_unassigned_threshold_085.joblib
pancreas_LR_balanced_level3_final_model_metadata_with_unassigned_threshold_085.json
pancreas_LR_balanced_level3_final_test_results_with_unassigned_threshold_085.xlsx
figures/
```

The final model bundle includes:

* trained balanced Logistic Regression model
* label encoder
* class names
* training HVG gene order
* gene symbols
* preprocessing parameters, including `target_sum`
* confidence threshold
* `Unassigned` label
* prediction output column information for backend integration

The test prediction table includes:

* true label
* predicted label before thresholding
* confidence score
* confidence threshold
* final label after thresholding
* whether the cell was labeled as `Unassigned`
* class probabilities for all pancreas Level 3 labels

## Final Pancreas Model with Unassigned Feature

After threshold analysis, the final pancreas model was updated to include a confidence-based `Unassigned` output. This feature prevents the model from forcing low-confidence cells into one of the known pancreas Level 3 classes.

