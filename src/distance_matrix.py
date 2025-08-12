import pickle
import shutil
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from __init__ import CACHE_DIR
from calculate_similarity import SimilarityMeasure


# this function calculates Distance Matrix or Similarity matrix
def DistanceMatrix(data: pd.DataFrame | None = None, uuid: UUID = uuid4()):
    """
    arguement
    ---------
    data - the dataset whose Similarity matrix we are going to calculate

    returns
    -------
    the distance matrix by loading th pickle file
    """

    if data is None:
        raise TypeError("data should be pd.DataFrame")

    pickleFilePath = Path(
        rf"{CACHE_DIR}\SimMat_{uuid}.pkl"
    )  # checking if the distance matrix was saved from last run to save processing

    if pickleFilePath.is_file():
        temp_file = open(pickleFilePath, "rb")
        return pickle.load(temp_file)

    Data_list = []
    for _, rows in data.iterrows():
        my_data = [rows.Hobby, rows.Age, rows.Educational_Level, rows.Marital_Status]
        Data_list.append(my_data)

    N = len(data)
    similarity_mat = np.zeros([N, N])  # for cosine np.ones
    for i in range(N):
        for j in range(N):
            similarity_mat[i][j] = SimilarityMeasure(Data_list[i], Data_list[j])

        with open(pickleFilePath, "wb") as file:
            pickle.dump(similarity_mat, file)

    temp_file = open(pickleFilePath, "rb")
    return pickle.load(temp_file)


if __name__ == "__main__":
    # for testing the module
    data = pd.read_csv("data/HAYES_ROTH.csv")
    data = data.drop(columns="Name")
    data = data.drop(columns="Class")
    dist_mat = DistanceMatrix(data)
    print(dist_mat.shape)
