import anndata as ad
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_split(file_path):
    adata = ad.read_h5ad(file_path)
    
    # Feature Matrix (Kept sparse by default to save RAM)
    X = adata.X
    
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Feature Names
    genes = adata.var_names.to_numpy()
    
    # Target 1: CMV Status (Binarized)
    y_cmv_raw = adata.obs["subject.cmv"].values
    y_cmv = np.where(y_cmv_raw == "Positive", 1, 0)
    
    # Target 2: Ethnicity
    y_eth = adata.obs["self_reported_ethnicity"].values
    
    # Target 3: Cell Type
    y_celltype = adata.obs["cell_type"].values
    
    # Target 4: Donor Level
    y_donor = adata.obs["donor_id"].values if "donor_id" in adata.obs else None
    
    return {
        "X": X,
        "genes": genes,
        "y_cmv": y_cmv,
        "y_eth": y_eth,
        "y_celltype": y_celltype,
        "y_donor": y_donor
    }
