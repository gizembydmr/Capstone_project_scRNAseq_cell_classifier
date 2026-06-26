# TISSUE-SPECIFIC CELL TYPE PREDICTION PLATFORM FROM SINGLE-CELL RNA SEQUENCING DATA USING MACHINE LEARNING TECHNIQUES

BAU Computer Engineering & Software Engineering Capstone Project

## Project Overview

This project is a tissue-specific cell type prediction platform for single-cell RNA sequencing (scRNA-seq) data. It combines preprocessing, supervised machine learning, inference-time gene alignment, visualization, differential gene expression analysis, SHAP-based explainability, and a graphical user interface into one accessible workflow.

The main value of the project is that it allows users to upload scRNA-seq data, run trained tissue-specific models, inspect predictions, visualize cell populations, and export biological interpretation outputs without needing to manually run multiple bioinformatics scripts.

## Final Platform

All necessary files required to run the final integrated application are located inside the `Final_Platform/` folder. This folder also includes the usage guide, final GUI/backend files, trained model files, requirements, and supporting components needed to launch and test the platform.

## Repository Structure and Task Distribution

- `src/ml_model/`, `src/Pancreas_Model/`, `src/preprocessing/`  
  **Feride Gizem Baydemir** developed the main machine learning workflow, including PBMC model training, model comparison, evaluation, confidence-based Unassigned thresholding, and final Logistic Regression model packaging. She also extended the platform to a second tissue by preparing the Tabula Sapiens pancreas dataset, creating the pancreas preprocessing and model development workflow, comparing models, and training the final pancreas model. In addition, she prepared the training preprocessing pipeline, including dataset validation, label hierarchy preparation, normalization, log transformation, highly variable gene selection, and training-ready `.h5ad` dataset creation.

- `src/Inference_Preprocessing/`  
  **Zeynep Ağmaz** developed the inference preprocessing and gene alignment workflow, ensuring that uploaded datasets are processed consistently with the training data and aligned to the saved model gene order. She also implemented the SHAP explainability workflow, including `explain_final_LR_with_SHAP.py`.

- `src/backend/`  
  **İrem Tuana Canarslan** developed the backend pipeline that connects file loading, validation, preprocessing, gene alignment, model prediction, visualization, differential gene expression analysis, and export functions into one complete workflow.

- `src/gui/`  
  **Nadira Yakupbayeva** developed the Streamlit graphical user interface, including dataset upload, tissue model selection, prediction execution, result display, model information pages, and export options.

- `src/visualization/`  
  **Gülsu Naz Koçak** developed the visualization and downstream analysis components, including PCA, UMAP, differential gene expression analysis, volcano plots, platform testing, and refinement of visualization outputs.

- `data/`  
  Contains project datasets, prepared test subsets, and data-related files used for model development, backend testing, and GUI demonstration.
