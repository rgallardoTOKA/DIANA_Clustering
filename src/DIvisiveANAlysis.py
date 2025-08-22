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

from src import CACHE_DIR
from utils.calculate_similarity import SimilarityMeasure, cosine, l2_norm


# this is the main class that computes the clusters using diana algorithm
@dataclass
class DianaClustering(BaseEstimator, ClusterMixin):
    n_clusters: int
    similarity_func: InitVar[str] = field(default="l2")
    _uuid: InitVar[UUID] = field(default=uuid4())
    data: pd.DataFrame = field(init=False)
    n_samples: int = field(init=False)
    n_features: int = field(init=False)
    cache_file: Path = field(init=False)
    similarity_matrix: NDArray[np.float64] = field(init=False)
    N: int = field(init=False)
    sim_func: Callable[[NDArray[np.float64], NDArray[np.float64]], np.floating] = field(
        init=False
    )
    data_list: list[Any] = field(init=False)

    def __post_init__(self, similarity_func, _uuid):
        """
        constructor of the class, it takes the main data frame as input
        """
        self.cache_file = Path(rf"{CACHE_DIR}\SimMat_{_uuid}.pkl")
        self.sim_func = l2_norm
        if similarity_func == "cosine":
            self.sim_func = cosine

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
    def __DistanceMatrix(self) -> NDArray[np.float64]:
        """
        arguement
        ---------
        data - the dataset whose Similarity matrix we are going to calculate

        returns
        -------
        the distance matrix by loading th pickle file
        """

        if self.cache_file.is_file():
            with open(self.cache_file, "rb") as f:
                temp_file = pickle.load(f)
            return temp_file

        self.data_list = self.data.values.tolist()

        # TODO: Refactor this
        similarity_mat = np.zeros(shape=[self.N, self.N])  # for cosine np.ones
        if self.sim_func == cosine:
            similarity_mat = np.ones(shape=[self.N, self.N])
        for i in range(self.N):
            for j in range(self.N):
                similarity_mat[i][j] = SimilarityMeasure(
                    np.array(self.data_list[i]),
                    np.array(self.data_list[j]),
                    self.sim_func,
                )

        with open(self.cache_file, "wb") as file:
            pickle.dump(similarity_mat, file)

        return similarity_mat

    def fit(self, X: pd.DataFrame):
        """
        this method uses the main Divisive Analysis algorithm to do the clustering

        arguements
        ----------
        n_clusters - integer number of clusters we want

        returns
        -------
        cluster_labels - numpy array an array where cluster number of a sample corrosponding to the same index is stored
        """
        self.data = X
        self.n_samples, self.n_features = self.data.shape
        self.N = self.data.shape[0]
        self.similarity_matrix = self.__DistanceMatrix()

        clusters = [
            list(range(self.n_samples))
        ]  # list of clusters, initially the whole dataset is a single cluster
        while True:
            c_diameters = [
                np.max(self.similarity_matrix[cluster][:, cluster])
                for cluster in clusters
            ]  # cluster diameters
            max_cluster_dia = np.argmax(c_diameters)  # maximum cluster diameter
            max_difference_index = np.argmax(
                np.mean(
                    self.similarity_matrix[clusters[max_cluster_dia]][
                        :, clusters[max_cluster_dia]
                    ],
                    axis=1,
                )
            )
            splinters = [
                clusters[max_cluster_dia][max_difference_index]
            ]  # spinter group
            last_clusters = clusters[max_cluster_dia]
            del last_clusters[max_difference_index]
            while True:
                split = False
                for j in range(len(last_clusters))[::-1]:
                    splinter_distances = self.similarity_matrix[
                        last_clusters[j], splinters
                    ]
                    last_distances = self.similarity_matrix[
                        last_clusters[j], np.delete(last_clusters, j, axis=0)
                    ]
                    if np.mean(splinter_distances) <= np.mean(last_distances):
                        splinters.append(last_clusters[j])
                        del last_clusters[j]
                        split = True
                        break
                if split is False:
                    break
            del clusters[max_cluster_dia]
            clusters.append(splinters)
            clusters.append(last_clusters)
            if len(clusters) == self.n_clusters:
                break

        cluster_labels = np.zeros(self.n_samples)
        for i in range(len(clusters)):
            cluster_labels[clusters[i]] = i

        self.labels_ = cluster_labels
        return self

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
