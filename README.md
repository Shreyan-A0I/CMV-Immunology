# CMV-Immunology: Single-Cell Classification

Analyzing the immune footprint of CMV infection at the single-cell level using the MESA cohort.

## Data Access
Download processed files (`train.h5ad`, `val.h5ad`) and place them in `processed_data/`:
[Google Drive Data Link](https://drive.google.com/drive/folders/1yEWdw0N5KekxKEDV4dY032IPcBguF2x0?usp=sharing)

## Analysis Methods

- **Method 1: Unsupervised Clustering**
  Uses PCA and UMAP to evaluate whether transcriptomic states can recover biological cell type identities without ground-truth labels.
- **Method 2: Logistic Regression**
  Regularized binary classification to identify single-cell gene induction patterns (e.g., *KLRD1*) that predict donor CMV status.
- **Method 3: Ethnicity Prediction**
  XGBoost classification to investigate demographic variance and ensure gene selection isn't biased by donor ethnicity.
- **Method 4 & 5: Advanced Modeling**
  Cascaded Bayesian priors and donor-level pseudobulking to refine classification accuracy and biological signal.

## How to Use

1. **Environment Setup**
   Requires Python 3.13.
   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Running Scripts**
   Execute methods directly from the root directory:
   ```bash
   python method1_clustering.py
   python method2_logistic_regression.py
   python method3_xgboost.py
   ```
   *Note: Results are saved to `plots/` and `results/`.*