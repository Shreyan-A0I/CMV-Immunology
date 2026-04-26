from dataloader import load_split
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (accuracy_score, classification_report, 
                           precision_score, recall_score, roc_auc_score,
                           roc_curve, auc, precision_recall_curve)
import xgboost as xgb
from itertools import cycle
from sklearn.utils.class_weight import compute_sample_weight

print("Loading data using centralized dataloader...")

try:
    # Load training data
    data = load_split("processed_data/train.h5ad")
    X = data["X"]
    y = data["y_eth"]
    
    if y is None:
        raise ValueError("Ethnicity labels not found in the dataset.")

    # Remove missing values if any
    valid_mask = pd.notna(y)
    X = X[valid_mask]
    y = y[valid_mask]

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    
    print(f"Data shape: {X.shape}")
    print(f"Found {n_classes} ethnicity groups: {list(le.classes_)}")

    # Split data (we split our train set further for internal validation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Create XGBoost model
    print("\nTraining XGBoost model for Ethnicity Prediction...")
    xgb_model = xgb.XGBClassifier(
        max_depth=6,
        learning_rate=0.3,
        n_estimators=50,  
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        objective='multi:softprob' if n_classes > 2 else 'binary:logistic'
    )
    
    # Handle class imbalance
    sample_weights = compute_sample_weight("balanced", y_train)
    
    # Train model
    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

    # Predictions
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)

    # 1. Classification Report
    print("\nEvaluation Results:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 2. Confusion Matrix Plot
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_test, y_pred)
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_perc, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.ylabel('Ground Truth')
    plt.xlabel('Predicted Ethnicity')
    plt.title('Ethnicity Prediction: Confusion Matrix (%)')
    plt.savefig("plots/method3_xgboost_confusion.png")
    plt.close()

    # 3. Multi-class ROC Curve
    plt.figure(figsize=(10, 8))
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    
    for i, class_name in enumerate(le.classes_):
        if n_classes == 2: # Binary case
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            break
        else:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-class ROC: Ethnicity Prediction')
    plt.legend(loc="lower right")
    plt.savefig("plots/method3_xgboost_roc.png")
    plt.close()

    print("\nResults and plots saved successfully.")

except Exception as e:
    print(f"Error: {e}")