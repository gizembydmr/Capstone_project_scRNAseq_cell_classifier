# ML Model Development

This folder contains the machine learning model comparison, final model training, and confidence-threshold analysis scripts for PBMC cell-type prediction.

## Scripts

1. `01_compare_logistic_regression.py`  
   Compares Logistic Regression models with and without class weighting using 5-fold stratified cross-validation.

2. `02_compare_random_forest.py`  
   Compares Random Forest models with and without class weighting using the same development/test split.

3. `03_compare_SVM.py`  
   Compares Linear SVM models with and without class weighting using the same development/test split.

4. `04_compare_all_models.py`  
   Combines the saved model comparison results into one summary table and figure set.

<img width="4497" height="1539" alt="table_pbmc_level1_model_comparison" src="https://github.com/user-attachments/assets/09541737-559a-426b-80ef-547427f56e76" />

5. `05_train_final_LR_model.py`  
   Trains the selected final model, Logistic Regression without class weighting, on the full development set and evaluates it on the untouched test set.

## Final saved model

`LR_level1_no_weight_final_model_bundle.joblib` contains:

- trained Logistic Regression model
- label encoder
- class names
- training HVG gene order
- gene symbols
- preprocessing parameters, including `target_sum`
- metadata required for inference integration

The inference pipeline should load this model bundle and use the saved preprocessing parameters and gene order.


## Level 2 Model Comparison

`06_compare_all_models_level2.py` compares PBMC68k cell-type prediction models using the intermediate label column `cell_type_level_2`.

It evaluates six configurations:

- Logistic Regression, with and without class weighting
- Linear SVM, with and without class weighting
- Random Forest, with and without class weighting

The script reuses the same 80/20 development-test split from the Level 1 Logistic Regression experiment. It performs 5-fold stratified cross-validation on the development set and evaluates each model once on the untouched 20% test set.

Use the 5-fold CV results for model comparison and model selection. Use the final test set results only as held-out confirmation.

**Input dataset:** `pbmc68k_preprocessed_for_training.h5ad`  
**Label column:** `cell_type_level_2`  
**Output folder:** `results/all_models_level2/`

Main outputs:

- `all_models_level2_CV_results.xlsx`
- `all_models_level2_metadata.json`
- `all_models_level2_split_indices.npz`
- CV and final-test confusion matrix heatmaps
- presentation summary table image
- Macro F1, Weighted F1, and Balanced Accuracy barplots

<img width="4497" height="1539" alt="table_pbmc_level2_model_comparison" src="https://github.com/user-attachments/assets/291e726c-0d41-45ae-aae1-398e99e2df77" />

## Level 3 Model Comparison

`07_compare_all_models_level3.py` compares PBMC68k cell-type prediction models using the fine-grained label column `cell_type_level_3`.

It evaluates six configurations:

- Logistic Regression, with and without class weighting
- Linear SVM, with and without class weighting
- Random Forest, with and without class weighting

The script reuses the same 80/20 development-test split from the Level 1 Logistic Regression experiment. It performs 5-fold stratified cross-validation on the development set and evaluates each model once on the untouched 20% test set.

Use the 5-fold CV results for model comparison and model selection. Use the final test set results only as held-out confirmation.

**Input dataset:** `pbmc68k_preprocessed_for_training.h5ad`  
**Label column:** `cell_type_level_3`  
**Output folder:** `results/all_models_level3/`

Main outputs:

- `all_models_level3_CV_results.xlsx`
- `all_models_level3_metadata.json`
- `all_models_level3_split_indices.npz`
- CV and final-test confusion matrix heatmaps
- presentation summary table image
- Macro F1, Weighted F1, and Balanced Accuracy barplots

<img width="4497" height="1539" alt="table_pbmc_level3_model_comparison" src="https://github.com/user-attachments/assets/009f975e-5d5d-419a-aac0-1da877d05eb3" />

## Level 3 PCA/SVD Experiment

`08_compare_all_models_level3_PCA.py` tests whether dimensionality reduction improves Level 3 cell-type prediction.

The script uses the same 2000-HVG input matrix and the same 80/20 development-test split as the previous comparisons, but adds a PCA-style dimensionality reduction step before model training. Since the dataset is large and sparse, TruncatedSVD is used as a memory-friendly PCA-like method.

It evaluates the same six configurations:

- Logistic Regression, with and without class weighting
- Linear SVM, with and without class weighting
- Random Forest, with and without class weighting

Both 50 and 200 components were tested. However, PCA/SVD did not improve performance; instead, the scores decreased compared with the direct 2000-HVG Level 3 models. This suggests that dimensionality reduction removed gene-level information that is important for distinguishing fine-grained cell subtypes.

### all models level 3 PCA presentation summary table 50 PC
<img width="5145" height="1539" alt="table_pbmc_level3_pca_50pc_model_comparison" src="https://github.com/user-attachments/assets/056154c2-2a39-481c-bac3-78485bb7b384" />

### all models level 3 PCA presentation summary table 200 PC
<img width="5145" height="1539" alt="table_pbmc_level3_pca_200pc_model_comparison" src="https://github.com/user-attachments/assets/ef67a156-9fbd-478a-b5cc-7f59916be642" />

Based on this experiment, the direct 2000-HVG representation was kept for further model development because it preserved more biological signal and remained more interpretable for gene-level analysis.

**Input dataset:** `pbmc68k_preprocessed_for_training.h5ad`  
**Label column:** `cell_type_level_3`  
**Output folder:** `results/all_models_level3_PCA/`

Main outputs:

- PCA/SVD model comparison Excel files
- metadata JSON file
- split index file
- explained variance tables/plots
- CV and final-test confusion matrix heatmaps
- presentation summary table image

## Level 3 Logistic Regression Model

`09_train_final_LR_model_level3.py` trains the final selected Level 3 Logistic Regression model using the fine-grained label column `cell_type_level_3`.

The model is trained on the full 80% development set and evaluated once on the untouched 20% test set, using the same split indices from the Level 1 experiment for consistency across all model comparisons.

**Input dataset:** `pbmc68k_preprocessed_for_training.h5ad`  
**Label column:** `cell_type_level_3`  
**Model:** Logistic Regression without class weighting  
**Output folder:** `results/final_LR_level3/`  
**Model folder:** `models/LR_level3/`

Main outputs:

- `LR_level3_no_weight_final_model_bundle.joblib`
- `LR_level3_no_weight_final_model_metadata.json`
- `LR_level3_no_weight_final_test_results.xlsx`
- final test confusion matrix tables and heatmap
- test prediction table with predicted labels and class probabilities

The saved model bundle includes the trained model, label encoder, class names, HVG gene order, gene symbols, and preprocessing parameters needed for inference integration.


## Level 3 Confidence Threshold Analysis

`10_compare_unassigned_thresholds_level3.py` evaluates confidence thresholds for adding an `Unassigned` output to the final Level 3 Logistic Regression model.

The script trains the same selected Logistic Regression model on the full 80% development set and evaluates different confidence thresholds on the untouched 20% test set. For each cell, the model calculates class probabilities using `predict_proba`, and the highest probability is used as the confidence score. If this confidence score is below the tested threshold, the cell is labeled as `Unassigned`.

**Input dataset:** `pbmc68k_preprocessed_for_training.h5ad`
**Label column:** `cell_type_level_3`
**Model:** Logistic Regression without class weighting
**Output folder:** `results/unassigned_thresholds_level3/`

Main outputs:

* `LR_level3_unassigned_threshold_comparison.xlsx`
* `LR_level3_unassigned_threshold_comparison_metadata.json`
* threshold summary table
* test predictions with confidence scores
* confusion matrices for each threshold
* presentation-ready threshold summary table figure
* threshold trade-off plot
* assigned vs unassigned cell percentage plot

<img width="3282" height="1957" alt="LR_level3_unassigned_threshold_tradeoff_plot" src="https://github.com/user-attachments/assets/c5c59e4f-c2d9-4a81-b2c1-b389b47f0555" />

<img width="4470" height="2669" alt="LR_level3_unassigned_threshold_summary_table" src="https://github.com/user-attachments/assets/24a6ecb6-3d43-4eba-99f0-53e15698d7d5" />

The tested thresholds ranged from 0.30 to 0.90. Increasing the threshold improved accuracy and Macro F1 among assigned cells, but also increased the number of cells labeled as Unassigned. Therefore, the threshold was not selected using accuracy or Macro F1 alone. Instead, a coverage-based selection rule was used: the threshold was selected as the highest tested threshold that preserved at least 95% assigned-cell coverage. This rule was chosen because the model is intended to be used as a practical annotation tool in the GUI, where most cells should still receive a useful predicted label, while the lowest-confidence predictions should still be flagged.

<img width="3264" height="1692" alt="LR_level3_unassigned_threshold_assigned_unassigned_plot" src="https://github.com/user-attachments/assets/54140be9-5002-4b54-a064-50bf76bbda93" />

Using this rule, a threshold of 0.50 was selected for the final PBMC Level 3 model. This was the highest tested threshold that kept assigned-cell coverage above 95%. At this threshold, approximately 95.21% of cells remained assigned and 4.79% were labeled as Unassigned. Accuracy on assigned cells was approximately 88.29%, and Macro F1 on assigned cells was approximately 76.53%. Higher thresholds gave better assigned-cell scores, but they reduced assigned-cell coverage below the selected 95% requirement.

## Final Level 3 Logistic Regression Model with Unassigned Feature

`11_train_final_LR_model_level3_with_unassigned_feature.py` trains the final selected Level 3 Logistic Regression model and saves it with the confidence-based `Unassigned` feature.

The model is trained on the full 80% development set and evaluated once on the untouched 20% test set, using the same split indices as the previous experiments. The final confidence threshold is set to **0.50**.

**Input dataset:** `pbmc68k_preprocessed_for_training.h5ad`
**Label column:** `cell_type_level_3`
**Model:** Logistic Regression without class weighting
**Confidence threshold:** `0.50`
**Unassigned label:** `Unassigned`
**Output folder:** `results/final_LR_unassigned_thresholds_level3/`

Main outputs:

* `LR_level3_no_weight_final_model_bundle_with_unassigned_threshold_050.joblib`
* `LR_level3_no_weight_final_model_metadata_with_unassigned_threshold_050.json`
* `LR_level3_no_weight_final_test_results_with_unassigned_threshold_050.xlsx`
* final test confusion matrix tables and heatmap
* test prediction table with confidence scores and final labels

The saved model bundle includes:

* trained Logistic Regression model
* label encoder
* class names
* training HVG gene order
* gene symbols
* preprocessing parameters, including `target_sum`
* confidence threshold
* `Unassigned` label
* prediction output columns for backend integration

During inference, the backend should first obtain the predicted class probabilities using `predict_proba`. The highest probability is stored as the confidence score. If this score is lower than `0.50`, the final prediction is returned as `Unassigned`; otherwise, the predicted Level 3 cell-type label is returned.

