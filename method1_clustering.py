# %%
from dataloader import load_split
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.cm as cm
from collections import Counter

# %%
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
import pandas as pd

# %%
def run_clustering_analysis(file_path):
    """
    Method 1: Presentation-ready clustering analysis (PC vs Raw).
    """
    # Load data
    data = load_split(file_path)
    X_all = data["X"]
    y_celltype_all = data["y_celltype"]
    
    print(f"Loaded data with shape: {X_all.shape}")
    
    def kmeans(X, k, max_iters=100, tol=1e-4):
        np.random.seed(42)
        n_samples, n_features = X.shape

        # Initialize centroids randomly
        random_idxs = np.random.choice(n_samples, k, replace=False)
        centroids = X[random_idxs]

        for i in range(max_iters):
            # Assign clusters
            distances = np.linalg.norm(X[:, None] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 else centroids[j]
                for j in range(k)
            ])

            # Check convergence
            if np.linalg.norm(new_centroids - centroids) < tol:
                print(f"Converged at iteration {i}")
                break

            centroids = new_centroids

        return labels, centroids

    if not isinstance(X_all, np.ndarray):
        X_all = X_all.toarray()

    # Increase subsample size for valid statistical comparison
    max_samples = 20000
    if X_all.shape[0] > max_samples:
        idx = np.random.choice(X_all.shape[0], max_samples, replace=False)
        X = X_all[idx]
        y_celltype = y_celltype_all[idx]
    else:
        X = X_all
        y_celltype = y_celltype_all

    k = len(np.unique(y_celltype))

    # Flow: PCA -> KMeans on PCs -> KMeans on Raw
    pca = PCA(n_components=25)
    X_pca = pca.fit_transform(X)
    X_2d = X_pca[:, :2] 

    print(f"Running K-Means on Top 25 PCs (k={k})")
    labels_pc, _ = kmeans(X_pca, k=k)

    print(f"Running K-Means on Raw 2000 Genes (k={k})")
    labels_raw, _ = kmeans(X, k=k)

    # Calculate ARI (Adjusted Rand Index) between PCs and Truth
    ari_score = adjusted_rand_score(y_celltype, labels_pc)
    print(f"Adjusted Rand Index (PC-Clustering vs Truth): {ari_score:.4f}")

    # Plot 1: Comparison Scatter (Presentation Style)
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    sns.set_style("white")
    cmap = cm.get_cmap('tab10', k)

    # Plot PC-based clusters
    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_pc, s=4, cmap=cmap, alpha=0.6)
    ax1.set_title("Feature Space: Top 25 Principal Components", fontsize=14, pad=15)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot Raw-based clusters
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_raw, s=4, cmap=cmap, alpha=0.6)
    ax2.set_title("Feature Space: Raw 2000 HVGs", fontsize=14, pad=15)
    ax2.set_xlabel("PC1")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add a vertical divider line between plots
    fig.subplots_adjust(wspace=0.1)
    trans = fig.transFigure
    line = plt.Line2D((0.5, 0.5), (0.1, 0.9), color="lightgrey", lw=2, transform=trans)
    fig.lines.append(line)

    plt.suptitle(f"K-Means Comparison: Dimensionality Reduction vs. Raw Features (k={k})", fontsize=16, y=1.02)
    plt.savefig("plots/method1_comparison_presentation.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: Heatmap Purity (Clusters vs Truth)
    plt.figure(figsize=(12, 8))
    
    # Create contingency table
    contingency_df = pd.DataFrame({'Cluster': labels_pc, 'Cell Type': y_celltype})
    heatmap_data = pd.crosstab(contingency_df['Cell Type'], contingency_df['Cluster'])
    
    # Normalize to percentages (row-wise)
    heatmap_perc = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

    sns.heatmap(heatmap_perc, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Percentage (%)'})
    plt.title(f"Clustering Purity Heatmap (K-Means on 25 PCs)\nAdjusted Rand Index (ARI) = {ari_score:.4f}", fontsize=15, pad=20)
    plt.xlabel("K-Means Cluster ID")
    plt.ylabel("Biological Cell Type (Ground Truth)")
    
    plt.tight_layout()
    plt.savefig("plots/method1_cluster_purity_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Output details for verification
    print("\nPC-based Cluster composition (top labels):")
    for cluster in np.unique(labels_pc):
        true_labels = y_celltype[labels_pc == cluster]
        top = Counter(true_labels).most_common(2)
        print(f"Cluster {cluster}: {top}")

# %%
if __name__ == "__main__":
    train_path = "processed_data/train.h5ad"
    run_clustering_analysis(train_path)

# %%