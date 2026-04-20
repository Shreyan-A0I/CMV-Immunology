import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from dataloader import load_split
from method2_logistic_regression import fit_logistic_regression, sigmoid, compute_log_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_cascaded_prior():
    print("Loading data...")
    train_data = load_split("processed_data/train.h5ad")
    val_data = load_split("processed_data/val.h5ad")
    
    X_train, y_train_cmv, y_train_eth = train_data["X"], train_data["y_cmv"].astype(int), train_data["y_eth"]
    X_val, y_val_cmv, y_val_eth = val_data["X"], val_data["y_cmv"].astype(int), val_data["y_eth"]

    # Filter NaNs in ethnicity manually to match XGBoost reqs
    valid_train = pd.notna(y_train_eth)
    X_train, y_train_cmv, y_train_eth = X_train[valid_train], y_train_cmv[valid_train], y_train_eth[valid_train]
    
    valid_val = pd.notna(y_val_eth)
    X_val, y_val_cmv, y_val_eth = X_val[valid_val], y_val_cmv[valid_val], y_val_eth[valid_val]

    le = LabelEncoder()
    y_train_eth_enc = le.fit_transform(y_train_eth)
    
    # 1. Calculate Bayesian Prior: P(CMV=1 | Ethnicity)
    print("\nCalculating Priors from Training Set...")
    priors = {}
    for i, eth_class in enumerate(le.classes_):
        class_mask = (y_train_eth_enc == i)
        if np.sum(class_mask) > 0:
            cmv_rate = np.mean(y_train_cmv[class_mask])
            priors[i] = cmv_rate
        else:
            priors[i] = np.mean(y_train_cmv) # global fallback
        print(f"  P(CMV=1 | Ethnicity={eth_class}) = {priors[i]:.4f}")

    # 2. Train XGBoost for Ethnicity
    print("\nTraining XGBoost Ethnicity Model...")
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train_eth_enc)
    
    xgb_model = xgb.XGBClassifier(
        max_depth=6, learning_rate=0.3, n_estimators=50, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_train_eth_enc, sample_weight=sample_weights)

    # 3. Predict max-confidence ethnicity
    print("Predicting ethnicity for assigning priors...")
    pred_eth_train = xgb_model.predict(X_train)
    pred_eth_val = xgb_model.predict(X_val)

    # 4. Map the predicted ethnicity to its prior likelihood
    prior_train = np.vectorize(priors.get)(pred_eth_train).reshape(-1, 1).astype(np.float32)
    prior_val = np.vectorize(priors.get)(pred_eth_val).reshape(-1, 1).astype(np.float32)

    # NOTE: The custom Gradient Descent LR model requires standardized inputs. 
    prior_mean = np.mean(prior_train)
    prior_std = np.std(prior_train) + 1e-8
    prior_train_scaled = (prior_train - prior_mean) / prior_std
    prior_val_scaled = (prior_val - prior_mean) / prior_std

    # 5. Combine features (Gene Expression + 1 Prior Feature)
    X_train_expanded = np.hstack([X_train, prior_train_scaled])
    X_val_expanded = np.hstack([X_val, prior_val_scaled])

    print("\nTraining Baseline Logistic Regression (Genes Only)...")
    lr = 0.01
    lam = 0.001
    iters = 1000
    weights_base, bias_base, loss_base = fit_logistic_regression(X_train, y_train_cmv, lr, lam, iters)
    
    print("\nTraining Cascaded Logistic Regression (Genes + Ethnicity Prior)...")
    weights_casc, bias_casc, loss_casc = fit_logistic_regression(X_train_expanded, y_train_cmv, lr, lam, iters)

    # 6. Evaluate
    def evaluate(X, weights, bias, y_true, name):
        scores = X @ weights + bias
        probs = sigmoid(scores)
        best_f1, best_thresh = 0, 0.5
        for t in np.arange(0.1, 0.95, 0.05):
            preds = (probs >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
        
        preds_best = (probs >= best_thresh).astype(int)
        roc = roc_auc_score(y_true, probs)
        prec = precision_score(y_true, preds_best, zero_division=0)
        rec = recall_score(y_true, preds_best, zero_division=0)
        print(f"[{name}] Optimal Thresh: {best_thresh:.2f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {best_f1:.4f} | ROC-AUC: {roc:.4f}")
        return probs

    print("\n--- Final Validation Results ---")
    _ = evaluate(X_val, weights_base, bias_base, y_val_cmv, "Baseline LR")
    _ = evaluate(X_val_expanded, weights_casc, bias_casc, y_val_cmv, "Cascaded LR")

    # 7. Plot Loss Curve Comparison
    plt.figure(figsize=(10, 6))
    plt.plot(loss_base, label="Baseline LR (Genes)", color="blue")
    plt.plot(loss_casc, label="Cascaded LR (Genes + Prior)", color="orange", linestyle="--")
    plt.title("Training Loss Progression: Baseline vs Cascaded Model")
    plt.xlabel("Iteration (x100)")
    plt.ylabel("Binary Cross Entropy Loss")
    plt.legend()
    plt.savefig("plots/method4_cascade_loss.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_cascaded_prior()
