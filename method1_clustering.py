# %%
from dataloader import load_split
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib.cm as cm
from collections import Counter

# %%
def run_clustering_analysis(file_path):
    """
    Method 1: Self-implemented clustering to see whether HVGs capture information right.
    """
    # Load data
    data = load_split(file_path)
    X = data["X"]
    y_celltype = data["y_celltype"]
    y_cmv = data["y_cmv"]
    
    print(f"Loaded data with shape: {X.shape}")
    
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

    if not isinstance(X, np.ndarray):
        X = X.toarray()

    X = np.log1p(X)

    max_samples = 5000
    if X.shape[0] > max_samples:
        idx = np.random.choice(X.shape[0], max_samples, replace=False)
        X = X[idx]
        y_celltype = y_celltype[idx]

    k = len(np.unique(y_celltype))

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    print(f"Running K-Means with k={k}")

    cluster_labels, centroids = kmeans(X, k=k)

    # Cluster vs True Label Alignment

    print("\nCluster composition (top labels):")
    cluster_summary = {}
    for cluster in np.unique(cluster_labels):
        true_labels = y_celltype[cluster_labels == cluster]
        top = Counter(true_labels).most_common(3)
        cluster_summary[cluster] = top
        print(f"Cluster {cluster}: {top}")

    # Plotting
    cmap = cm.get_cmap('tab10', k)

    plt.figure()
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, s=5, cmap=cmap)
    plt.title("K-Means Clusters (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()

    # Final comparison with true labels

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_celltype)

    cmap = cm.get_cmap('tab10', len(le.classes_))

    plt.figure()
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_encoded, s=5, cmap=cmap)
    plt.title("True Cell Types (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    # Legend with correct colors
    handles = []
    for i, label in enumerate(le.classes_):
        handles.append(
            plt.Line2D([], [], marker='o', linestyle='', color=cmap(i), label=label)
        )
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()

# %%
if __name__ == "__main__":
    train_path = "processed_data/train.h5ad"
    run_clustering_analysis(train_path)

# %%