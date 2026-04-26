# CMV-Immunology: Single-Cell Classification

Analyzing the cell types and ethnicity effect on predicting CMV infection given gene expression.

## Data Access
Download processed files (`train.h5ad`, `val.h5ad`,`test.h5ad`) and place them in `processed_data/`:
[Google Drive Data Link](https://drive.google.com/drive/folders/1yEWdw0N5KekxKEDV4dY032IPcBguF2x0?usp=sharing)

## Analysis Methods

- **Method 1: Unsupervised Clustering**
  Uses PCA+kmeans evaluate whether we can recover biological cell type identities without ground-truth labels, and see which variance in cell types by their ability to cluster together.
- **Method 2: Logistic Regression**
  L2 regularized binary classification to predict donor CMV status.
- **Method 3: Ethnicity Prediction**
  XGBoost classification to investigate demographic variance and ensure gene selection isn't biased by donor ethnicity.
- **Method 4 & 5: Supporting methods**
  addition of Bayesian priors and donor-level pseudobulking to refine classification accuracy and biological signal for CMV prediciton in method 2.

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
