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
