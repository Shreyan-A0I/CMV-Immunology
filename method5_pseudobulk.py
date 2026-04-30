import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from dataloader import load_split

# Evaluates the pseudobulk performance of the logistic regression and XGBoost models 
def run_pseudobulk_evaluation():
    train_data = load_split("processed_data/train.h5ad")
    val_data = load_split("processed_data/val.h5ad")
    
    X_train, y_train_cmv, y_train_eth = train_data["X"], train_data["y_cmv"], train_data["y_eth"]
    X_val, y_val_cmv, y_val_eth = val_data["X"], val_data["y_cmv"], val_data["y_eth"]
    y_val_donor = val_data["y_donor"]

    if y_val_donor is None:
        raise ValueError("y_donor missing from validation data.")

    # Clean missing ethnicity labels
    train_eth_mask = pd.notna(y_train_eth)
    X_train_eth, y_train_eth_clean = X_train[train_eth_mask], y_train_eth[train_eth_mask]
    
    val_eth_mask = pd.notna(y_val_eth)
    X_val_eth, y_val_eth_clean = X_val[val_eth_mask], y_val_eth[val_eth_mask]
    y_val_donor_eth = y_val_donor[val_eth_mask]

    le = LabelEncoder()
    y_train_eth_enc = le.fit_transform(y_train_eth_clean)
    y_val_eth_enc = le.transform(y_val_eth_clean)

def run_pseudobulk_evaluation():
    train_data = load_split("processed_data/train.h5ad")
    val_data = load_split("processed_data/val.h5ad")
    
    X_train, y_train_cmv, y_train_eth = train_data["X"], train_data["y_cmv"], train_data["y_eth"]
    X_val, y_val_cmv, y_val_eth = val_data["X"], val_data["y_cmv"], val_data["y_eth"]
    y_val_donor = val_data["y_donor"]

    if y_val_donor is None:
        raise ValueError("y_donor missing from validation data.")

    # Ethnicity filter
    train_eth_mask = pd.notna(y_train_eth)
    X_train_eth, y_train_eth_clean = X_train[train_eth_mask], y_train_eth[train_eth_mask]
    
    val_eth_mask = pd.notna(y_val_eth)
    X_val_eth, y_val_eth_clean = X_val[val_eth_mask], y_val_eth[val_eth_mask]
    y_val_donor_eth = y_val_donor[val_eth_mask]

    le = LabelEncoder()
    y_train_eth_enc = le.fit_transform(y_train_eth_clean)
    y_val_eth_enc = le.transform(y_val_eth_clean)

    # Models
    print("Training models...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train_cmv)

    sample_weights = compute_sample_weight("balanced", y_train_eth_enc)
    xgb_model = xgb.XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=50, random_state=42, n_jobs=-1)
    xgb_model.fit(X_train_eth, y_train_eth_enc, sample_weight=sample_weights)

    # Predicting
    cmv_probs = lr_model.predict_proba(X_val)[:, 1]
    eth_probs = xgb_model.predict_proba(X_val_eth)

    # Aggregate
    print("\nDonor-level summary:")
    
    # CMV
    cmv_df = pd.DataFrame({'donor': y_val_donor, 'true': y_val_cmv, 'prob': cmv_probs})
    cmv_agg = cmv_df.groupby('donor').agg({'true': 'first', 'prob': 'mean'})
    cmv_pred = (cmv_agg['prob'] >= 0.5).astype(int)
    
    print(f"\nCMV (n={len(cmv_agg)})")
    print(f"Accuracy: {accuracy_score(cmv_agg['true'], cmv_pred):.4f}")
    print(classification_report(cmv_agg['true'], cmv_pred, target_names=["Neg", "Pos"], zero_division=0))

    # Ethnicity
    eth_df = pd.DataFrame(eth_probs)
    eth_df['donor'] = y_val_donor_eth
    eth_df['true'] = y_val_eth_enc
    
    eth_agg = eth_df.groupby('donor').mean()
    eth_true = eth_agg['true'].astype(int)
    eth_pred = eth_agg.drop(columns=['true']).values.argmax(axis=1)

    print(f"\nEthnicity (n={len(eth_agg)})")
    print(f"Accuracy: {accuracy_score(eth_true, eth_pred):.4f}")
    print(classification_report(eth_true, eth_pred, labels=np.arange(len(le.classes_)), target_names=le.classes_, zero_division=0))

if __name__ == "__main__":
    run_pseudobulk_evaluation()