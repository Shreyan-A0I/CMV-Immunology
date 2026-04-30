# %%
from dataloader import load_split
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter
import umap
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from matplotlib.lines import Line2D

# %%
def run_clustering_analysis(file_path):
    # Load
    data = load_split(file_path)
    X_all = data["X"]
    y_celltype_all = data["y_celltype"]
    
    print(f"Data shape: {X_all.shape}")
    
    def kmeans(X, k, max_iters=100, tol=1e-4):
        np.random.seed(42)
        n_samples, n_features = X.shape
        random_idxs = np.random.choice(n_samples, k, replace=False)
        centroids = X[random_idxs]
        for i in range(max_iters):
            distances = np.linalg.norm(X[:, None] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.array([
                X[labels == j].mean(axis=0) if len(X[labels == j]) > 0 else centroids[j]
                for j in range(k)
            ])
            if np.linalg.norm(new_centroids - centroids) < tol:
                print(f"Done at iter {i}")
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

    unique_types = np.unique(y_celltype)
    k = len(unique_types)

    # Dim Reduction
    print("Running PCA (25)...")
    pca = PCA(n_components=25)
    X_pca = pca.fit_transform(X)
    
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_pca)

    print(f"K-Means (25 PCs)...")
    labels_pc, _ = kmeans(X_pca, k=k)

    print(f"K-Means (Raw Genes)...")
    labels_raw, _ = kmeans(X, k=k)

    # Label Alignment (Majority Voting)
    def align_labels(labels, true_labels):
        mapped_labels = np.empty_like(true_labels, dtype=object)
        for cluster in np.unique(labels):
            mask = (labels == cluster)
            # Find most common true label in this cluster
            majority_type = Counter(true_labels[mask]).most_common(1)[0][0]
            mapped_labels[mask] = majority_type
        return mapped_labels

    aligned_pc = align_labels(labels_pc, y_celltype)
    aligned_raw = align_labels(labels_raw, y_celltype)

    ari_score = adjusted_rand_score(y_celltype, labels_pc)
    print(f"ARI (PC vs Truth): {ari_score:.4f}")

    # Consistent Coloring 
    palette = sns.color_palette("tab10", k)
    type_to_color = {t: palette[i] for i, t in enumerate(unique_types)}
    
    def get_colors(labels):
        return [type_to_color[l] for l in labels]

    # 3-Panel Plot
    plt.close('all')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
    sns.set_style("white")
    
    # Plotting helper
    def plot_umap(ax, labels, title):
        cols = get_colors(labels)
        ax.scatter(X_umap[:, 0], X_umap[:, 1], c=cols, s=3, alpha=0.5)
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("UMAP1")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plot_umap(ax1, y_celltype, "Biological Truth (Cell Type)")
    ax1.set_ylabel("UMAP2")
    plot_umap(ax2, aligned_pc, f"K-Means (25 PCs)\nARI = {ari_score:.4f}")
    plot_umap(ax3, aligned_raw, "K-Means (Raw 2000 Genes)")

    # Legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=t,
                             markerfacecolor=c, markersize=8) for t, c in type_to_color.items()]
    ax3.legend(handles=legend_elements, title="Cell Types", loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)

    plt.suptitle(f"Consistent Comparison: Ground Truth vs. Mathematical Clusters", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig("plots/method1_umap_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Scree Plot & Heatmap
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, 26), pca.explained_variance_ratio_ * 100, color='royalblue', alpha=0.7)
    plt.title("Scree Plot")
    plt.savefig("plots/method1_scree_plot.png")
    plt.close()

    contingency_df = pd.DataFrame({'Cluster': labels_pc, 'Cell Type': y_celltype})
    heatmap_data = pd.crosstab(contingency_df['Cell Type'], contingency_df['Cluster'])
    heatmap_perc = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_perc, annot=True, fmt=".1f", cmap="YlGnBu")
    plt.title("Cluster Purity Heatmap (Mapped)")
    plt.savefig("plots/method1_cluster_purity_heatmap.png")
    plt.close()

    print("\nAligned Cluster Mapping:")
    for cluster in np.unique(labels_pc):
        majority = Counter(y_celltype[labels_pc == cluster]).most_common(1)[0][0]
        print(f"Cluster {cluster} -> {majority}")

if __name__ == "__main__":
    train_path = "processed_data/train.h5ad"
    run_clustering_analysis(train_path)