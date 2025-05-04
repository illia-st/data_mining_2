from collections import deque
import numpy as np

class DBSCAN:
    UNVISITED_LABEL = -2
    NOISE_LABEL = -1

    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts
        self.labels = []

    def fit(self, X):
        cluster_id = 0
        points_cnt = X.shape[0]
        self.labels = np.full(points_cnt, self.UNVISITED_LABEL)

        for point_idx in range(points_cnt):
            if self.labels[point_idx] != self.UNVISITED_LABEL:
                continue

            neighbors_idx = self.__find_neighbors(X, point_idx)

            if len(neighbors_idx) < self.min_pts:
                self.labels[point_idx] = self.NOISE_LABEL
            else:
                self.__expand_cluster(X, point_idx, neighbors_idx, cluster_id)
                cluster_id += 1
        
        return self.labels

    def __find_neighbors(self, X, point_idx):
        neighbors = []
        points_cnt = X.shape[0]
        point = X[point_idx]

        for i in range(points_cnt):
            if self.__distance(point, X[i]) < self.eps:
                neighbors.append(i)

        return neighbors

    def __distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def __expand_cluster(self, X, point_idx, point_neighbors, cluster_id):
        self.labels[point_idx] = cluster_id
        queue = deque(point_neighbors)

        while queue:
            cur_neighbor_idx = queue.popleft()

            if self.labels[cur_neighbor_idx] == self.NOISE_LABEL:
                self.labels[cur_neighbor_idx] = cluster_id
            elif self.labels[cur_neighbor_idx] == self.UNVISITED_LABEL:
                self.labels[cur_neighbor_idx] = cluster_id
                cur_point_neighbors = self.__find_neighbors(X, cur_neighbor_idx)

                if len(cur_point_neighbors) >= self.min_pts:
                    queue.extend(cur_point_neighbors)