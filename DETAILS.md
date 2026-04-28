# CMV-Immunology: Methodological Details & Results

This document provides a deep dive into the computational methods, biological rationale, and performance metrics for the CMV status classification pipeline.

---

## Analysis Methods

### Method 1: Unsupervised Clustering & Manifold Analysis
- **Why:** To verify if biological cell types can be recovered from gene expression alone without using CMV labels. This ensures our feature selection captures fundamental biology rather than technical noise.
- **What:** PCA-based denoising followed by K-Means clustering and UMAP visualization.
- **How:** We reduce the 2000 highly variable genes (HVGs) to 25 principal components, then apply K-Means (k=6) to find clusters. Labels are aligned to ground truth using majority voting for consistent coloring.
- **Results:** 
    - **ARI:** ~0.42 (Significant overlap with biological cell types).
    - **Variance Explained:** 25 PCs explain ~13.5% of total variance.

### Method 2: Optimized Logistic Regression (Baseline)
- **Why:** To identify a specific transcriptomic "footprint" of CMV infection across thousands of cells.
- **What:** Binary classification using regularized Logistic Regression.
- **How:** Implemented from scratch with **L2 Regularization ($\lambda=0.001$)**, a **learning rate of 0.1**, and **balanced class weights** to handle the CMV+/- donor imbalance.
- **Results:**
    - **ROC-AUC:** 0.6652
    - **Clinical Screening Balance:** At a threshold of 0.75, precision reaches ~0.65. At 0.10, recall is ~0.97.

### Method 3: Ethnicity Prediction (Bias Control)
- **Why:** To determine if donor ethnicity is a confounding factor that could "leak" into our CMV classification.
- **What:** Multi-class classification of donor demographics.
- **How:** Training an XGBoost model on gene expression to predict 5 ethnicity groups (African American, Asian, Hispanic, etc.).
- **Results:** The model successfully identifies ethnicity from expression, highlighting the need to account for demographic skews in clinical models.

### Method 4: Cascaded Bayesian Prior
- **Why:** To test if incorporating demographic CMV prevalence ($P(CMV|Ethnicity)$) improves single-cell classification.
- **What:** A two-stage model that injects an ethnicity-based prior into the LR model.
- **How:** An XGBoost model predicts ethnicity; the corresponding population-level CMV probability is then appended as a feature to the Gene-only LR model.
- **Results:** 
    - **ROC-AUC:** 0.6592 (Slight drop vs. Gene-only).
    - **Conclusion:** The gene expression data already "encodes" the relevant demographic signal; adding the prior explicitly is redundant.

### Method 5: Pseudobulk Evaluation
- **Why:** To assess performance at the donor level by aggregating single-cell signals.
- **What:** Summing gene counts per donor to create a "pseudobulk" profile.
- **How:** Aggregating all cells for a donor and applying the trained LR weights.
- **Results:**
    - **Accuracy:** ~53%.
    - **Conclusion:** Single-cell heterogeneity is high; donor-level aggregation currently dilutes the subtle CMV-specific signal.

---

## Gene Importance (Biological Signal)
By analyzing the weights from Method 2, we identified the key drivers of the CMV classification:

| Gene Symbol | Weight | Status | Biological Significance |
| :--- | :--- | :--- | :--- |
| **KLRD1 (CD94)** | +0.221 | CMV+ | Critical receptor for adaptive NK/T-cell expansion in CMV. |
| **MT-ND2** | +0.254 | CMV+ | High metabolic activity in inflationary T-cell subsets. |
| **EGR1** | -0.160 | CMV- | Marker for naive/homeostatic cells; decreased in chronic infection. |

---

## Visualization & Plots
All generated plots are saved in the `plots/` directory:
- `method1_umap_comparison.png`: 3-panel comparison of Truth vs. PC-Clustering vs. Raw-Clustering.
- `method1_scree_plot.png`: Individual and cumulative variance explained by PCs.
- `method1_cluster_purity_heatmap.png`: Matrix showing how well clusters match cell types.
- `method2_performance.png`: ROC and Precision-Recall curves.
- `method3_xgboost_roc.png`: Demographic prediction performance.
- `method3_xgboost_confusion.png`: Ethnicity classification error matrix.
- `gene_importance.png`: Bar chart of the top 20 genes (Positive vs. Negative contributors).
