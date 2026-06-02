# ML Model Development

This folder contains the machine learning model comparison and final model training scripts for Level 1 PBMC cell-type prediction.

## Scripts

1. `01_compare_logistic_regression.py`  
   Compares Logistic Regression models with and without class weighting using 5-fold stratified cross-validation.

2. `02_compare_random_forest.py`  
   Compares Random Forest models with and without class weighting using the same development/test split.

3. `03_compare_SVM.py`  
   Compares Linear SVM models with and without class weighting using the same development/test split.

4. `04_compare_all_models.py`  
   Combines the saved model comparison results into one summary table and figure set.

<img width="4470" height="1228" alt="all_models_level1_summary_table" src="https://github.com/user-attachments/assets/150a7c54-da50-4683-b8e7-ff7416e44353" />

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

<img width="3870" height="1106" alt="all_models_level2_presentation_summary_table" src="https://github.com/user-attachments/assets/d1df4abb-2e17-48f4-95cd-25898e92aea1" />

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

<img width="3870" height="1106" alt="all_models_level3_presentation_summary_table" src="https://github.com/user-attachments/assets/19eb6cbe-30e7-4fe4-b2f8-b137cd02ee4e" />

