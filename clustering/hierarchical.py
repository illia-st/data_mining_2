import numpy as np
from scipy.spatial.distance import pdist, squareform
import heapq

class AgglomerativeClustering:
    def __init__(self, n_clusters=3, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels = []

    def fit(self, X):
        self.X = X
        self.n_samples = X.shape[0]
        self.clusters = {i: [i] for i in range(self.n_samples)}
        self.cluster_sizes = {i: 1 for i in range(self.n_samples)}

        self.distance_matrix = squareform(pdist(X, metric='euclidean'))
        np.fill_diagonal(self.distance_matrix, np.inf)

        self.heap = self.__initialize_heap()

        while len(self.clusters) > self.n_clusters:
            _, cluster_i, cluster_j = heapq.heappop(self.heap)

            if cluster_i not in self.clusters or cluster_j not in self.clusters:
                continue

            self.__merge_clusters(cluster_i, cluster_j)

        return self.__finalize_labels()

    def __initialize_heap(self):
        heap = []
        for i in range(self.n_samples):
            for j in range(i+1, self.n_samples):
                dist = self.__evaluate_distance(self.clusters[i], self.clusters[j])
                heapq.heappush(heap, (dist, i, j))
        return heap

    def __merge_clusters(self, i, j):
        new_cluster = self.clusters[i] + self.clusters[j]
        new_cluster_id = max(self.clusters.keys()) + 1

        self.clusters[new_cluster_id] = new_cluster
        self.cluster_sizes[new_cluster_id] = len(new_cluster)

        del self.clusters[i]
        del self.clusters[j]

        for cluster_id in self.clusters:
            if cluster_id != new_cluster_id:
                dist = self.__evaluate_distance(self.clusters[new_cluster_id], self.clusters[cluster_id])
                heapq.heappush(self.heap, (dist, new_cluster_id, cluster_id))

    def __evaluate_distance(self, cluster_a, cluster_b):
        points_a = self.X[cluster_a]
        points_b = self.X[cluster_b]

        if self.linkage == 'single':
            return np.min(np.linalg.norm(points_a[:, None] - points_b, axis=2))
        elif self.linkage == 'complete':
            return np.max(np.linalg.norm(points_a[:, None] - points_b, axis=2))
        elif self.linkage == 'average':
            return np.mean(np.linalg.norm(points_a[:, None] - points_b, axis=2))
        elif self.linkage == 'ward':
            centroid_a = np.mean(points_a, axis=0)
            centroid_b = np.mean(points_b, axis=0)
            size_a, size_b = len(points_a), len(points_b)
            return (size_a * size_b) / (size_a + size_b) * np.sum((centroid_a - centroid_b) ** 2)
        else:
            raise ValueError(f"Invalid linkage method: {self.linkage}")

    def __finalize_labels(self):
        labels = np.zeros(self.n_samples, dtype=int)
        for label, cluster in enumerate(self.clusters.values()):
            labels[cluster] = label
        self.labels = labels
        return labels
