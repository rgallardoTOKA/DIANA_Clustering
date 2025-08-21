# import numpy as np
import pandas as pd

from src.DIvisiveANAlysis import DianaClustering


def main():
    data = pd.read_csv("data/HAYES_ROTH.csv")  # reading the dataset csv file
    data = data.drop(columns="Name")  # droping the unnecessary features
    data = data.drop(columns="Class")

    diana = DianaClustering(n_clusters=3)
    # clusters = diana.fit_predict(
    diana.fit(
        data
    )  # as there is 3 classes we chose to divide the dataset in 3 clusters  # applying the Diana Clustering algorithm

    # np.save("test/data/example_classes.npy", clusters)
    # np.save("test/data/example_simmilarity.npy", diana.similarity_matrix)
    test_data = pd.DataFrame([[2, 1, 3, 2], [2, 4, 5, 6]])
    prediction = diana.predict(test_data)
    test_data["predictions"] = prediction
    print(test_data)
    print(diana.labels_)


if __name__ == "__main__":
    main()
