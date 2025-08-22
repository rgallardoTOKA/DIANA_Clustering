from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, ClusterMixin, check_is_fitted
from sklearn.utils.validation import validate_data  # type: ignore


# this is the main class that computes the clusters using diana algorithm
@dataclass
class DianaClustering(BaseEstimator, ClusterMixin):
    n_clusters: int
    similarity_func: str = field(default="euclidean")
    data: ArrayLike = field(init=False)
    n_samples: int = field(init=False)
    similarity_matrix: NDArray[np.float64] = field(init=False)

    def __check_test_data(self, X):
        X = validate_data(
            self,
            X,
            accept_sparse="csr",
            reset=False,
            dtype=[np.float64, np.float32],
            order="C",
            accept_large_sparse=False,
        )
        return X

    def __split_cluster(self, cluster: list[int]):
        # Find the pair of points with the largest distance
        sub_matrix = self.similarity_matrix[
            np.ix_(cluster, cluster)
        ]  # Sub-matrix for the current cluster
        farthest_points = np.unravel_index(np.argmax(sub_matrix), sub_matrix.shape)

        # Create two subclusters
        point1, point2 = cluster[farthest_points[0]], cluster[farthest_points[1]]
        cluster1 = [point1]
        cluster2 = [point2]

        # Assign points to the closer of the two farthest points
        for point in cluster:
            if point != point1 and point != point2:
                dist_to_cluster1 = self.similarity_matrix[point, point1]
                dist_to_cluster2 = self.similarity_matrix[point, point2]
                if dist_to_cluster1 < dist_to_cluster2:
                    cluster1.append(point)
                else:
                    cluster2.append(point)

        return cluster1, cluster2

    def fit(self, X: ArrayLike):
        self.data = X
        self.n_samples = self.data.shape[0]  # type: ignore
        self.similarity_matrix = squareform(pdist(X, metric=self.similarity_func))  # type: ignore

        clusters = [
            list(range(self.n_samples))
        ]  # list of clusters, initially the whole dataset is a single cluster

        while len(clusters) < self.n_clusters:
            # Find the largest cluster to split
            largest_cluster = max(clusters, key=len)
            clusters.remove(largest_cluster)

            # Split the largest cluster
            cluster1, cluster2 = self.__split_cluster(largest_cluster)
            clusters.append(cluster1)
            clusters.append(cluster2)

        cluster_labels = np.zeros(self.n_samples)
        for i in range(len(clusters)):
            cluster_labels[clusters[i]] = i

        self.labels_ = cluster_labels
        return self

    def predict(self, data: ArrayLike):
        check_is_fitted(self)
        X = self.__check_test_data(data)
        sim_measures = cdist(X, self.data, metric=self.similarity_func)  # type: ignore
        index_of_cluster = np.argmin(sim_measures, axis=1)
        return self.labels_[index_of_cluster]
