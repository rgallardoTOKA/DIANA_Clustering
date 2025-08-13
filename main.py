import pandas as pd

from src.DIvisiveANAlysis import DianaClustering


def main():
    data = pd.read_csv("data/HAYES_ROTH.csv")  # reading the dataset csv file
    data = data.drop(columns="Name")  # droping the unnecessary features
    data = data.drop(columns="Class")

    diana = DianaClustering(data)  # applying the Diana Clustering algorithm
    clusters = diana.fit(
        3
    )  # as there is 3 classes we chose to divide the dataset in 3 clusters
    print(clusters)


if __name__ == "__main__":
    main()
