# CMV-Immunology: Single-Cell Pipeline for CMV Status Classification

This repository contains a comprehensive pipeline for analyzing single-cell RNA-seq data from the Multi-Ethnic Study of Atherosclerosis (MESA) cohort, specifically focused on classifying Cytomegalovirus (CMV) infection status.

## Data Access
The processed single-cell data used in this project can be downloaded from the following link:
[Processed Data (Google Drive)](https://drive.google.com/drive/folders/1yEWdw0N5KekxKEDV4dY032IPcBguF2x0?usp=sharing)

Please place the `train.h5ad` and `val.h5ad` files inside a `processed_data/` directory in the root of this project.

## Methods Overview

### Method 1: Unsupervised Clustering
*   **Why:** To determine if the transcriptomic states of cells alone are sufficient to recover biological identity (cell types) without the use of ground truth labels. This acts as a quality control for our feature selection.
*   **What:** We use PCA for initial denoising, followed by UMAP for non-linear dimensionality reduction. Clustering is performed via K-Means on the principal components and visualized on the UMAP manifold to evaluate Adjusted Rand Index (ARI) against known cell types.

### Method 2: Logistic Regression (Baseline Model)
*   **Why:** To establish a predictive baseline for identifying the immune "footprint" of CMV status. We aim to find a specific gene induction pattern that distinguishes CMV+ donors at the single-cell level.
*   **What:** A regularized (L2) Logistic Regression model with balanced class weights. It provides a highly interpretable set of feature weights, allowing us to identify specific "driver" genes (e.g., *KLRD1*) associated with chronic viral infection.

### Method 3: Ethnicity Prediction (Demographic Control)
*   **Why:** To investigate demographic skews and verify if the 2000 highly variable genes capture donor ethnicity. This is critical for identifying potential skews or "shortcuts" in our CMV classification model.
*   **What:** A Gradient Boosting (XGBoost) model trained to predict donor ethnicity. This model helps us understand the baseline transcriptomic variance explained by genetic background rather than disease status.

## How to Use

### 1. Environment Setup
This project requires Python `3.13`. We recommend using a virtual environment:

```bash
# Create and activate virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Pipeline
You can run each method as a standalone script. Ensure your data is in `processed_data/` before running:

```bash
# Run clustering analysis
python method1_clustering.py

# Run baseline classification
python method2_logistic_regression.py

# Run ethnicity prediction
python method3_xgboost.py

# Run advanced experiments (Cascaded Prior / Pseudobulking)
python method4_cascade.py
python method5_pseudobulk.py
```

### 3. Interpreting Results
- Plots are saved to the `plots/` directory.
- Model weights and CSV reports are saved to the `results/` or the root directory.
- Use `gene_importance.py` to extract and visualize the top genes contributing to the LR model.