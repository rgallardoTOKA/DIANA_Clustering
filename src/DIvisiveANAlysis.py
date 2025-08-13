import os
import pickle
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src import CACHE_DIR
from utils.calculate_similarity import SimilarityMeasure


# this is the main class that computes the clusters using diana algorithm
@dataclass
class DianaClustering:
    data: pd.DataFrame
    n_samples: int = field(init=False)
    n_features: int = field(init=False)
    uuid: UUID = InitVar[uuid4()]
    cache_file: Path = field(init=False)
    similarity_matrix: NDArray[np.float64] = field(init=False)
    N: int = field(init=False)

    def __post_init__(self):
        """
        constructor of the class, it takes the main data frame as input
        """
        self.n_samples, self.n_features = self.data.shape
        self.cache_file = Path(rf"{CACHE_DIR}\SimMat_{self.uuid}.pkl")
        self.N = self.data.shape[0]
        self.similarity_matrix = self.DistanceMatrix()

    # this function calculates Distance Matrix or Similarity matrix
    def DistanceMatrix(self) -> NDArray[np.float64]:
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

        Data_list = self.data.values.tolist()

        # TODO: Refactor this
        similarity_mat = np.zeros([self.N, self.N])  # for cosine np.ones
        for i in range(self.N):
            for j in range(self.N):
                similarity_mat[i][j] = SimilarityMeasure(
                    np.array(Data_list[i]), np.array(Data_list[j])
                )

        with open(self.cache_file, "wb") as file:
            pickle.dump(similarity_mat, file)

        return similarity_mat

    def fit(self, n_clusters):
        """
        this method uses the main Divisive Analysis algorithm to do the clustering

        arguements
        ----------
        n_clusters - integer number of clusters we want

        returns
        -------
        cluster_labels - numpy array an array where cluster number of a sample corrosponding to the same index is stored
        """
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
            if len(clusters) == n_clusters:
                break

        cluster_labels = np.zeros(self.n_samples)
        for i in range(len(clusters)):
            cluster_labels[clusters[i]] = i

        return cluster_labels

    # TODO: Implement this method
    def predict(self, data):
        pass

    def __del__(self):
        os.remove(rf"{CACHE_DIR}\SimMat_{self.uuid}.pkl")
