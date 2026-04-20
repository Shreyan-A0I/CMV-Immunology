import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (accuracy_score, classification_report, 
                           precision_score, recall_score, roc_auc_score,
                           roc_curve, auc, precision_recall_curve)
import xgboost as xgb
from itertools import cycle

#configure scanpy
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

print("Loading train.h5ad file (this may take a while for large files)...")

try:
    #load h5ad
    adata = sc.read_h5ad("train.h5ad")
    
    #use gene expression
    #convert to dense
    if adata.shape[0] > 0:
        X = adata.X.todense() if hasattr(adata.X, 'todense') else adata.X
        y_data = adata.obs
    else:
        X = adata.X.todense() if hasattr(adata.X, 'todense') else adata.X
        y_data = adata.obs
    
    X = np.array(X)
    
    #find ethnicity columns
    ethnicity_cols = ['ethnicity', 'race', 'ancestry', 'subject.race', 'subject.ethnicity', 'self_reported_ethnicity']
    ethnicity_col = None
    
    for col in ethnicity_cols:
        if col in y_data.columns:
            ethnicity_col = col
            break
    
    #search ethnicity keywords
    # i didn't actually remember the column so I just searched these terms lol
    if not ethnicity_col:
        for col in y_data.columns:
            if any(keyword in col.lower() for keyword in ['ethnicity', 'race']):
                ethnicity_col = col
                print(f"Found potential ethnicity column: {col}")
                break
    
    if ethnicity_col:
        y = y_data[ethnicity_col].values
        
        #remove missing values
        valid_mask = pd.notna(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        #encode labels
        if y.dtype == 'object' or isinstance(y[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        unique_ethnicities, counts = np.unique(y, return_counts=True)
        
        #split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        #create xgboost
        print("\nTraining XGBoost model")
        xgb_model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.3,
            n_estimators=50,  
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1 
        )
        
        #train model
        xgb_model.fit(X_train, y_train)
        
        #make predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)
        
        #evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nEthnicity Prediction Accuracy: {accuracy:.4f}")
        
        #calculate precision recall
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\nDetailed Evaluation Metrics:")
        print(f"Precision (Macro Average): {precision_macro:.4f}")
        print(f"Precision (Weighted Average): {precision_weighted:.4f}")
        print(f"Recall (Macro Average): {recall_macro:.4f}")
        print(f"Recall (Weighted Average): {recall_weighted:.4f}")
        
        #calculate roc
        #binarize output
        y_test_bin = label_binarize(y_test, classes=range(len(unique_ethnicities)))
        
        #calculate auc
        try:
            roc_auc_ovo = roc_auc_score(y_test, y_pred_proba, multi_class='ovo', average='macro')
            roc_auc_ovr = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
            print(f"ROC-AUC (One-vs-One): {roc_auc_ovo:.4f}")
            print(f"ROC-AUC (One-vs-Rest): {roc_auc_ovr:.4f}")
        except Exception as e:
            print(f"ROC-AUC no work: {e}")
        
        #print report
        if 'le' in locals():
            target_names = le.classes_
        else:
            target_names = [f"Ethnicity_{i}" for i in range(len(unique_ethnicities))]
        
        print("\nEthnicity Classification metrics:")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        
        #plot roc
        plt.figure(figsize=(15, 10))
        
        #compute roc
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(unique_ethnicities)):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        #plot curves
        plt.subplot(2, 2, 1)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'red'])
        for i, color in zip(range(len(unique_ethnicities)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'{target_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Ethnicity Classification')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        #plot precision
        plt.subplot(2, 2, 2)
        for i, color in zip(range(len(unique_ethnicities)), colors):
            precision_curve, recall_curve, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
            pr_auc = auc(recall_curve, precision_curve)
            plt.plot(recall_curve, precision_curve, color=color, lw=2,
                     label=f'{target_names[i]} (AUC = {pr_auc:.2f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Ethnicity Classification')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        #plot confusion
        from sklearn.metrics import confusion_matrix
        plt.subplot(2, 2, 3)
        cm = confusion_matrix(y_test, y_pred)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        #add annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        #plot distribution
        plt.subplot(2, 2, 4)
        class_counts = np.bincount(y_test)
        plt.bar(range(len(target_names)), class_counts, color=['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'red'][:len(target_names)])
        plt.xlabel('Ethnicity Classes')
        plt.ylabel('Number of Samples')
        plt.title('Test Set Class Distribution')
        plt.xticks(range(len(target_names)), target_names, rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        #individual performance
        print("\nIndividual Class Performance:")
        for i, class_name in enumerate(target_names):
            class_precision = precision_score(y_test == i, y_pred == i, zero_division=0)
            class_recall = recall_score(y_test == i, y_pred == i, zero_division=0)
            print(f"{class_name:<20} - Precision: {class_precision:.4f}, Recall: {class_recall:.4f}, ROC-AUC: {roc_auc[i]:.4f}")
        
        #show importance
        print("\nTop 10 most important genes for ethnicity prediction:")
        feature_importance = xgb_model.feature_importances_
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        
        for idx in top_features_idx:
            print(f"  {adata.var_names[idx]}: {feature_importance[idx]:.4f}")
except Exception as e:
    print(f"Error loading or processing data: {e}")