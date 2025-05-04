import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, tol=1e-4, random_state=None, centroids=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = centroids

    def _initialize_centroids(self, X):
        if self.centroids is not None:
            return self.centroids
        if self.random_state is not None:
            np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        return np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else X[np.random.randint(0, X.shape[0])]
            for i in range(self.n_clusters)
        ])

    def _has_converged(self, old_centroids, new_centroids):
        return np.all(np.linalg.norm(old_centroids - new_centroids, axis=1) < self.tol)

    def fit(self, X):
        self.centroids = self._initialize_centroids(X)
        for _ in range(self.max_iters):
            labels = self._assign_clusters(X, self.centroids)
            new_centroids = self._update_centroids(X, labels)
            if self._has_converged(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        return labels, self.centroids

    def predict(self, X):
        return self._assign_clusters(X, self.centroids)