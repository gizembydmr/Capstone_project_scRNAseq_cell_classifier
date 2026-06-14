# Visualization Module

## Overview

This module contains dimensionality reduction and visualization analyses performed on the PBMC68K single-cell RNA sequencing (scRNA-seq) dataset.

The objective is to explore cell population structure, evaluate dimensionality reduction techniques, and visualize relationships between major immune cell populations.

---

## Files

### `pca_umap.py`

Performs dimensionality reduction and visualization using:

- Principal Component Analysis (PCA)
- Uniform Manifold Approximation and Projection (UMAP)

Generated figures:

- `pca_cell_type.png`
- `pca_cell_type_level_1.png`
- `umap_cell_type.png`
- `umap_cell_type_level_1.png`

---

### `pca_variance_analysis.py`

Analyzes the variance explained by principal components.

Generated figures:

- `pca_variance_ratio.png`
- `pca_cumulative_variance.png`

These plots were used to determine an appropriate number of principal components for downstream analyses.

---

### `tsne.py`

Applies t-distributed Stochastic Neighbor Embedding (t-SNE) as an additional dimensionality reduction technique.

Generated figure:

- `tsne_cell_type_level_1.png`

The resulting t-SNE embedding was compared with UMAP to evaluate cluster separation and overall cell population structure.

---

## Methods

### Principal Component Analysis (PCA)

PCA reduces the dimensionality of high-dimensional gene expression data while preserving the largest sources of variance.

It serves as the initial dimensionality reduction step before applying non-linear visualization methods.

### UMAP

UMAP projects cells into a two-dimensional space while preserving both local and global data structure.

This visualization enables clear identification of major immune cell populations, including:

- T cells
- B cells
- NK cells
- Monocytes
- Dendritic cells
- Progenitor cells

### t-SNE (Additional Analysis)

t-SNE was explored as an additional visualization approach.

While t-SNE successfully separated major cell populations, UMAP provided a clearer representation of global cellular relationships and was therefore selected as the primary visualization method.

---

## Generated Figures

### PCA

<img src="https://raw.githubusercontent.com/gizembydmr/Capstone_project_scRNAseq_cell_classifier/main/src/visualization/figures/pca_cell_type_level_1.png" width="700">

### PCA Variance Ratio

<img src="https://raw.githubusercontent.com/gizembydmr/Capstone_project_scRNAseq_cell_classifier/main/src/visualization/figures/pca_variance_ratio.png" width="700">

### PCA Cumulative Variance

<img src="https://raw.githubusercontent.com/gizembydmr/Capstone_project_scRNAseq_cell_classifier/main/src/visualization/figures/pca_cumulative_variance.png" width="700">

### UMAP (Primary Visualization)

<img src="https://raw.githubusercontent.com/gizembydmr/Capstone_project_scRNAseq_cell_classifier/main/src/visualization/figures/umap_cell_type_level_1.png" width="700">

### Additional Analysis: t-SNE

<img src="https://raw.githubusercontent.com/gizembydmr/Capstone_project_scRNAseq_cell_classifier/main/src/visualization/figures/tsne_cell_type_level_1.png" width="700">

---

## Key Findings

- PCA effectively reduced dimensionality while preserving the major sources of variance.
- Variance analysis supported the selection of principal components for downstream analyses.
- UMAP clearly separated major PBMC cell populations.
- Increasing the number of principal components from 30 to 50 did not significantly improve cluster separation.
- t-SNE produced clustering patterns similar to UMAP and was included as an additional analysis.
- UMAP was selected as the primary visualization technique due to its clearer representation of overall cellular relationships.

---

## Technologies Used

- Python
- Scanpy
- AnnData
- NumPy
- Pandas
- Matplotlib
