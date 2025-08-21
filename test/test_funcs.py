import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.DIvisiveANAlysis import DianaClustering


# / Testing DIANA
@pytest.fixture
def diana():
    return DianaClustering(n_clusters=3)


@pytest.fixture
def sample_dataframe():
    return pd.read_csv("test/data/HAYES_ROTH_Modified.csv")


@pytest.fixture
def sample_classes():
    return np.load("test/data/example_classes.npy")


@pytest.fixture
def sample_sim_matrix():
    return np.load("test/data/example_simmilarity.npy")


def test_fit(diana, sample_classes, sample_dataframe):
    np.testing.assert_array_equal(diana.fit_predict(X=sample_dataframe), sample_classes)


def test_sim_matrix_shape(diana, sample_dataframe):
    diana.fit(sample_dataframe)
    assert diana.similarity_matrix.shape == (132, 132)


def test_sim_matrix(diana, sample_sim_matrix, sample_dataframe):
    diana.fit(sample_dataframe)
    np.testing.assert_array_equal(diana.similarity_matrix, sample_sim_matrix)


def test_predict_simple(diana, sample_dataframe):
    diana.fit(sample_dataframe)
    assert diana.predict([[2, 1, 3, 2]]) == [0.0]


def test_predict_multiple(diana, sample_dataframe):
    diana.fit(sample_dataframe)
    np.testing.assert_array_equal(
        diana.predict([[2, 1, 3, 2], [2, 4, 5, 6]]), [0.0, 0.0]
    )


def test_no_fit_predict(diana):
    with pytest.raises(NotFittedError):
        diana.predict([[2, 1, 3, 2]])


def test_not_correct_structure_predict(diana, sample_dataframe):
    diana.fit(sample_dataframe)
    print(diana.n_features_in_)
    # with pytest.raises(NotFittedError):
    diana.predict([[2, 1, 3]])
