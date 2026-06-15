# TISSUE-SPECIFIC CELL TYPE PREDICTION PLATFORM FROM SINGLE-CELL RNA SEQUENCING DATA USING MACHINE LEARNING TECHNIQUES


## Overview

CellPredict is an end-to-end single-cell RNA sequencing (scRNA-seq) analysis platform designed for automated cell type prediction, visualization, and differential gene expression analysis.

The platform provides a user-friendly graphical interface built with Streamlit and enables researchers to upload scRNA-seq datasets, perform automated cell type classification using pre-trained machine learning models, explore cellular embeddings through dimensionality reduction techniques, and identify differentially expressed genes between predicted cell populations.

---

## Features

### Cell Type Prediction

- Automated cell type classification using pre-trained Logistic Regression models
- Hierarchical prediction workflow
- Prediction confidence estimation
- Multi-class cell type annotation

### Data Validation

- Validation of uploaded `.h5ad` datasets
- Dataset integrity checks
- Gene compatibility verification
- Informative error reporting

### Visualization

- UMAP embeddings
- PCA embeddings
- Interactive visual exploration
- PNG export functionality

### Differential Gene Expression (DGE)

- Pairwise differential expression analysis
- Wilcoxon rank-sum statistical testing
- Volcano plot visualization
- Significant gene annotation
- CSV export of DGE results

### Export Options

- Prediction results (CSV)
- DGE results (CSV)
- UMAP plots (PNG)
- PCA plots (PNG)

---

## Project Structure

```text
Final_Platform/
│
├── backend.py
├── gui.py
├── pipeline.py
├── dge.py
├── data_loader.py
├── data_validation.py
├── gene_alignment.py
├── preprocess_inference.py
├── pca_umap.py
│
├── models/
│   ├── LR_level1_no_weight_final_model.pkl
│   ├── LR_level2_no_weight_final_model.pkl
│   └── LR_level3_no_weight_final_model.pkl
│
├── data/
│   └── test_files/
│
├── figures/
├── results/
├── requirements.txt
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone <repository_url>
cd Final_Platform
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

#### macOS / Linux

```bash
source venv/bin/activate
```

#### Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

- Python 3.11+
- Scanpy
- AnnData
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Streamlit
- adjustText

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Streamlit interface:

```bash
streamlit run gui.py
```

After startup, the application will be available at:

```text
http://localhost:8501
```

---

## Supported Input Format

### Accepted File Type

```text
.h5ad
```

### Required Dataset Contents

The uploaded AnnData object should contain:

- Gene expression matrix
- Gene identifiers
- Cell metadata

The platform automatically validates the uploaded dataset before analysis begins.

---

## Usage Guide

### Step 1 – Upload Dataset

Upload a valid `.h5ad` file using the upload panel.

The platform automatically performs:

- File validation
- Dataset integrity checks
- Gene compatibility verification

### Step 2 – Run Cell Type Prediction

Click **Run Prediction**.

The prediction pipeline performs:

1. Data loading
2. Dataset validation
3. Preprocessing
4. Gene alignment
5. Feature preparation
6. Cell type prediction
7. Probability estimation

### Step 3 – Explore Embeddings

After prediction completes, visualization tabs become available.

#### UMAP

UMAP provides a low-dimensional representation of cellular relationships and predicted cell populations.

#### PCA

PCA provides a principal component representation of the dataset structure.

Both visualizations can be exported as PNG images.

### Step 4 – Differential Gene Expression Analysis

Navigate to the DGE section.

Select two predicted cell types and click **Run DGE**.

The platform performs:

- Pairwise differential expression analysis
- Wilcoxon rank-sum testing
- Log fold-change calculation
- Adjusted p-value calculation
- Volcano plot generation
- Significant gene annotation

### Step 5 – Export Results

Available exports include:

- Prediction results (CSV)
- DGE results (CSV)
- UMAP plots (PNG)
- PCA plots (PNG)

---

## Differential Gene Expression Workflow

The DGE module:

1. Subsets cells belonging to selected cell types
2. Performs differential expression analysis using Scanpy
3. Maps Ensembl identifiers to gene symbols
4. Calculates log fold changes and adjusted p-values
5. Identifies significant genes
6. Generates annotated volcano plots
7. Produces downloadable result tables

---

## Machine Learning Workflow

```text
Input Dataset
      │
      ▼
Data Validation
      │
      ▼
Preprocessing
      │
      ▼
Gene Alignment
      │
      ▼
Feature Selection
      │
      ▼
Logistic Regression Model
      │
      ▼
Cell Type Predictions
      │
      ▼
Visualization & DGE
```

---
### Example Test Datasets

Example `.h5ad` files are provided in:

data/test_files/

These datasets can be used to test the complete prediction, visualization, and DGE workflow without requiring external data preparation.

---
## Error Handling

The platform includes validation and exception handling for:

- Invalid file formats
- Missing datasets
- Corrupted AnnData objects
- Missing required metadata
- Unsupported gene sets
- Model loading failures
- Invalid DGE comparisons
- Missing prediction results

All errors are reported directly through the graphical user interface.

---

## Testing and Validation

The platform was tested using multiple integration and user-interface test scenarios, including:

- Dataset validation
- Prediction workflow execution
- UMAP generation
- PCA generation
- Differential gene expression analysis
- Volcano plot visualization
- Download functionality
- Session reset and dataset replacement workflows

Identified issues were documented and resolved through iterative integration testing.

---

## Authors

Developed as part of the capstone project:

**Tissue-Specific Cell Type Prediction Platform from Single-Cell RNA Sequencing Data Using Machine Learning Techniques**

Bahçeşehir University (BAU)  
Department of Computer Engineering & Software Engineering  
Capstone Project

---

## License

This project is intended for academic and research purposes only.