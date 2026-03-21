import anndata as ad
import numpy as np

def load_split(file_path):
    """
    Loads a processed .h5ad file and returns a dictionary of arrays 
    for custom scratch-built models.
    """
    adata = ad.read_h5ad(file_path)
    
    # Feature Matrix (Kept sparse by default to save RAM)
    # Teammates can call .toarray() on this if their custom math needs dense arrays
    X = adata.X
    
    # Feature Names
    genes = adata.var_names.to_numpy()
    
    # Target 1: CMV Status (Binarized)
    y_cmv_raw = adata.obs["subject.cmv"].values
    y_cmv = np.where(y_cmv_raw == "Positive", 1, 0)
    
    # Target 2: Ethnicity
    y_eth = adata.obs["self_reported_ethnicity"].values
    
    # Target 3: Cell Type
    y_celltype = adata.obs["cell_type"].values
    
    return {
        "X": X,
        "genes": genes,
        "y_cmv": y_cmv,
        "y_eth": y_eth,
        "y_celltype": y_celltype
    }
