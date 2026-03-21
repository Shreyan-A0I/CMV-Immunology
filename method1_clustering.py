from dataloader import load_split
import numpy as np

def run_clustering_analysis(file_path):
    """
    Method 1: Self-implemented clustering to see whether HVGs capture information right.
    """
    # Load data
    data = load_split(file_path)
    X = data["X"]
    y_celltype = data["y_celltype"]
    y_cmv = data["y_cmv"]
    
    # Preprocessing (e.g., converting to dense if needed for custom math)
    # X_dense = X.toarray()
    
    print(f"Loaded data with shape: {X.shape}")
    
    # TODO: Implement custom clustering algorithm from scratch
    # Goals:
    # 1. Cluster the cells/samples.
    # 2. Evaluate if clusters correspond to biological signal (cell type, CMV status).
    # 3. Verify if High Variable Genes (HVGs) are driving the correct separation.
    
    pass

if __name__ == "__main__":
    train_path = "processed_data/train.h5ad"
    run_clustering_analysis(train_path)
