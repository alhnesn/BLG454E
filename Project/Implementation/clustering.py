import numpy as np
from sklearn.metrics import silhouette_score

def k_means_clustering(self):
    self.error_label.config(text="")
    if len(self.data) < 2:
        self.error_label.config(text="Error: Not enough data points")
        return

    try:
        n_clusters = int(self.kmeans_entry.get())
    except ValueError:
        n_clusters = None

    x = np.array(self.data)
    if n_clusters is None:
        n_clusters = find_optimal_k(self, x)
        if n_clusters is None:
            self.error_label.config(text="Error: Cannot determine optimal k for the given data")
            return

    if n_clusters >= len(x):
        self.error_label.config(text="Error: Number of clusters must be less than the number of data points")
        return

    try:
        if self.kmeans_plot:
            self.kmeans_plot.remove()
        if self.kmeans_centers_plot:
            self.kmeans_centers_plot.remove()

        # Perform K-Means clustering manually
        centroids, labels = k_means_manual(x, n_clusters)
        
        self.kmeans_plot = self.ax.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap='viridis', label='Clusters', zorder=2)
        self.kmeans_centers_plot = self.ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=100, alpha=0.75, marker='x', label='Centroids', zorder=3)
        self.ax.legend()
        self.canvas.draw()
    except Exception as e:
        self.error_label.config(text=f"Error: {str(e)}")

def k_means_manual(data, n_clusters, max_iter=100):
    np.random.seed(42)
    centroids = data[np.random.choice(data.shape[0], n_clusters, replace=False)]
    for _ in range(max_iter):
        labels = np.array([np.argmin([np.dot(x - c, x - c) for c in centroids]) for x in data])
        new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(n_clusters)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

def find_optimal_k(self, data):
    silhouette_scores = []
    k_range = range(2, min(len(data), 10))
    for k in k_range:
        centroids, labels = k_means_manual(data, k)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    if not silhouette_scores:
        return None
    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    return optimal_k

def remove_kmeans_clustering(self):
    if self.kmeans_plot:
        self.kmeans_plot.remove()
        self.kmeans_plot = None
    if self.kmeans_centers_plot:
        self.kmeans_centers_plot.remove()
        self.kmeans_centers_plot = None
    self.kmeans = None
    handles, labels = self.ax.get_legend_handles_labels()
    new_handles_labels = [(h, l) for h, l in zip(handles, labels) if 'Clusters' not in l and 'Centroids' not in l]
    if new_handles_labels:
        handles, labels = zip(*new_handles_labels)
        self.ax.legend(handles=handles, labels=labels)
    else:
        if self.ax.get_legend():
            self.ax.get_legend().remove()
    self.canvas.draw()

def agglomerative_clustering(self):
    self.error_label.config(text="")
    if len(self.data) < 2:
        self.error_label.config(text="Error: Not enough data points")
        return

    try:
        n_clusters = int(self.agglom_entry.get())
    except ValueError:
        n_clusters = None

    x = np.array(self.data)
    if n_clusters is None:
        n_clusters = find_optimal_k_agglomerative(self, x)
        if n_clusters is None:
            self.error_label.config(text="Error: Cannot determine optimal k for the given data")
            return

    if n_clusters >= len(x):
        self.error_label.config(text="Error: Number of clusters must be less than the number of data points")
        return

    try:
        if self.kmeans_centers_plot:  # Remove K-Means centroids if they exist
            self.kmeans_centers_plot.remove()
            self.kmeans_centers_plot = None

        if self.agglom_plot:
            self.agglom_plot.remove()

        labels = agglomerative_clustering_manual(x, n_clusters)

        self.agglom_plot = self.ax.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap='viridis', label='Clusters', zorder=2)
        self.ax.legend()
        self.canvas.draw()
    except Exception as e:
        self.error_label.config(text=f"Error: {str(e)}")

def agglomerative_clustering_manual(data, n_clusters):
    from scipy.cluster.hierarchy import linkage, fcluster
    Z = linkage(data, 'ward')
    labels = fcluster(Z, n_clusters, criterion='maxclust')
    return labels - 1  # Adjusting labels to start from 0

def find_optimal_k_agglomerative(self, data):
    silhouette_scores = []
    k_range = range(2, min(len(data), 10))
    for k in k_range:
        labels = agglomerative_clustering_manual(data, k)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    if not silhouette_scores:
        return None
    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    return optimal_k

def remove_agglomerative_clustering(self):
    if self.agglom_plot:
        self.agglom_plot.remove()
        self.agglom_plot = None
    self.agglom = None
    handles, labels = self.ax.get_legend_handles_labels()
    new_handles_labels = [(h, l) for h, l in zip(handles, labels) if 'Clusters' not in l]
    if new_handles_labels:
        handles, labels = zip(*new_handles_labels)
        self.ax.legend(handles=handles, labels=labels)
    else:
        if self.ax.get_legend():
            self.ax.get_legend().remove()
    self.canvas.draw()
