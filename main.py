import numpy as np
from sklearn.datasets import make_blobs

from src.DIvisiveANAlysis import DianaClustering


def main():
    # data = pd.read_csv("data/HAYES_ROTH.csv")  # reading the dataset csv file
    # data = data.drop(columns="Name")  # droping the unnecessary features
    # data = data.drop(columns="Class")
    X, y = make_blobs(n_samples=800, n_features=50, centers=3, random_state=20020906)  # type: ignore
    diana = DianaClustering(n_clusters=3)
    clusters = diana.fit_predict(
        # diana.fit(
        X
    )  # as there is 3 classes we chose to divide the dataset in 3 clusters  # applying the Diana Clustering algorithm

    X_test, y_train = make_blobs( # type: ignore
        n_samples=200, n_features=50, centers=3, random_state=20020906
    )
    X_unique, _ = make_blobs( # type: ignore
        n_samples=1, n_features=50, centers=3, random_state=20020906
    )
    prediction = diana.predict(X_test)
    print(diana.predict(X_unique))
    np.save("test/data/example_classes.npy", clusters)
    np.save("test/data/example_simmilarity.npy", diana.similarity_matrix)
    np.save("test/data/example_test_classes.npy", prediction)


if __name__ == "__main__":
    main()
