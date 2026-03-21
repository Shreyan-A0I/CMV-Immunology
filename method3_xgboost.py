from dataloader import load_split
import xgboost as xgb
import pandas as pd
import numpy as np

def run_xgboost_analysis():
    """
    Method 3: XGBoost to predict ethnicity and correlate with CMV susceptibility (using package).
    """
    # Load data
    train_data = load_split("processed_data/train.h5ad")
    
    X_train = train_data["X"]
    y_eth_train = train_data["y_eth"]
    y_cmv_train = train_data["y_cmv"]
    gene_names = train_data["genes"]

    print(f"Running XGBoost analysis for ethnicity prediction...")

    # TODO: 
    # 1. Train XGBoost classifier to predict y_eth from X.
    # 2. Extract feature importance to see which genes drive ethnicity differences.
    # 3. Correlate ethnicity-driving genes with CMV susceptibility results from Method 2.
    # 4. Analyze if certain ethnicities are more susceptible based on gene expression patterns.

    pass

if __name__ == "__main__":
    run_xgboost_analysis()
