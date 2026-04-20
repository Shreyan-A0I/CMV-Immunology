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
    # Load data
    data = load_split(file_path)
    X_all = data["X"]
    y_celltype_all = data["y_celltype"]
    
    print(f"Data shape: {X_all.shape}")
    
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

    # Subsample for plotting
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

    print(f"K-Means (25 PCs)...")
    labels_pc, _ = kmeans(X_pca, k=k)

    print(f"K-Means (Raw Genes)...")
    labels_raw, _ = kmeans(X, k=k)

    ari_score = adjusted_rand_score(y_celltype, labels_pc)
    print(f"ARI (PC vs Truth): {ari_score:.4f}")

    # Plot Comparison
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    sns.set_style("white")
    cmap = cm.get_cmap('tab10', k)

    ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_pc, s=4, cmap=cmap, alpha=0.6)
    ax1.set_title("Clusters: Top 25 PCs", fontsize=14, pad=15)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_raw, s=4, cmap=cmap, alpha=0.6)
    ax2.set_title("Feature Space: Raw 2000 HVGs", fontsize=14, pad=15)
    ax2.set_xlabel("PC1")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.subplots_adjust(wspace=0.1)
    trans = fig.transFigure
    line = plt.Line2D((0.5, 0.5), (0.1, 0.9), color="lightgrey", lw=2, transform=trans)
    fig.lines.append(line)

    plt.suptitle(f"Clustering Comparison (k={k})", fontsize=16, y=1.02)
    plt.savefig("plots/method1_comparison_presentation.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Heatmap Purity
    plt.figure(figsize=(12, 8))
    contingency_df = pd.DataFrame({'Cluster': labels_pc, 'Cell Type': y_celltype})
    heatmap_data = pd.crosstab(contingency_df['Cell Type'], contingency_df['Cluster'])
    heatmap_perc = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100

    sns.heatmap(heatmap_perc, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': '%'})
    plt.title(f"Cluster Purity (ARI = {ari_score:.4f})", fontsize=15, pad=20)
    plt.xlabel("Cluster ID")
    plt.ylabel("Ground Truth")
    
    plt.tight_layout()
    plt.savefig("plots/method1_cluster_purity_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\nCluster breakdown:")
    for cluster in np.unique(labels_pc):
        true_labels = y_celltype[labels_pc == cluster]
        top = Counter(true_labels).most_common(2)
        print(f"ID {cluster}: {top}")

# %%
if __name__ == "__main__":
    train_path = "processed_data/train.h5ad"
    run_clustering_analysis(train_path)

# %%