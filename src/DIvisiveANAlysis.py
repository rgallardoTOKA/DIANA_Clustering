import os
import pickle
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Callable
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClusterMixin, check_is_fitted
from sklearn.utils.validation import validate_data  # type: ignore
from scipy.spatial.distance import pdist, squareform

from src import CACHE_DIR
from utils.calculate_similarity import SimilarityMeasure, cosine, l2_norm


# this is the main class that computes the clusters using diana algorithm
@dataclass
class DianaClustering(BaseEstimator, ClusterMixin):
    n_clusters: int
    similarity_func: str = field(default="euclidean")
    _uuid: InitVar[UUID] = field(default=uuid4())
    data: pd.DataFrame = field(init=False)
    n_samples: int = field(init=False)
    n_features: int = field(init=False)
    cache_file: Path = field(init=False)
    similarity_matrix: NDArray[np.float64] = field(init=False)
    N: int = field(init=False)
    data_list: list[Any] = field(init=False)

    def __post_init__(self, _uuid):
        """
        constructor of the class, it takes the main data frame as input
        """
        self.cache_file = Path(rf"{CACHE_DIR}\SimMat_{_uuid}.pkl")

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

    # this function calculates Distance Matrix or Similarity matrix
    def __DistanceMatrix(self, data) -> NDArray[np.float64]:
        similarity_mat = squareform(pdist(data, metric=self.similarity_func)) # type: ignore
        return similarity_mat

    def __split_cluster(self, cluster):
        # Find the pair of points with the largest distance
        sub_matrix = self.similarity_matrix[np.ix_(cluster, cluster)]  # Sub-matrix for the current cluster
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

    def fit(self, X: pd.DataFrame):
        self.data = X
        self.n_samples, self.n_features = self.data.shape
        self.N = self.data.shape[0]
        self.similarity_matrix = self.__DistanceMatrix(self.data)

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

    # TODO Redo this function
    def __predict_value(self, data: NDArray[np.float64]):
        sim_measures = np.zeros(shape=[self.N])
        if self.sim_func == cosine:
            sim_measures = np.ones(shape=[self.N])
        for i in range(self.N):
            sim_measures[i] = SimilarityMeasure(data, self.data_list[i], self.sim_func)

        index_of_cluster = np.argmin(sim_measures)
        return float(self.labels_[index_of_cluster])

    def predict(self, data: list[list[Any]]):
        check_is_fitted(self)
        X = self.__check_test_data(data)
        return [self.__predict_value(x) for x in X]

    def __del__(self):
        os.remove(self.cache_file)
