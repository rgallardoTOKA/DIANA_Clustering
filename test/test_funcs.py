import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.exceptions import NotFittedError

from src.DIvisiveANAlysis import DianaClustering


# / Testing DIANA
@pytest.fixture
def diana():
    return DianaClustering(n_clusters=3)


@pytest.fixture
def sample_train_data():
    return make_blobs(n_samples=800, n_features=50, centers=3, random_state=20020906)[0]  # type: ignore


@pytest.fixture
def sample_test_data():
    return make_blobs(n_samples=200, n_features=50, centers=3, random_state=20020906)[0]  # type: ignore


@pytest.fixture
def sample_test_unique():
    return make_blobs(n_samples=1, n_features=50, centers=3, random_state=20020906)[0]  # type: ignore


@pytest.fixture
def sample_train_classes():
    return np.load("test/data/example_classes.npy")


@pytest.fixture
def sample_test_classes():
    return np.load("test/data/example_test_classes.npy")


@pytest.fixture
def sample_sim_matrix():
    return np.load("test/data/example_simmilarity.npy")


def test_fit(diana, sample_train_classes, sample_train_data):
    np.testing.assert_array_equal(
        diana.fit_predict(X=sample_train_data), sample_train_classes
    )


def test_sim_matrix_shape(diana, sample_train_data):
    diana.fit(sample_train_data)
    assert diana.similarity_matrix.shape == (800, 800)


def test_sim_matrix(diana, sample_sim_matrix, sample_train_data):
    diana.fit(sample_train_data)
    np.testing.assert_array_equal(diana.similarity_matrix, sample_sim_matrix)


def test_predict_simple(diana, sample_train_data, sample_test_unique):
    diana.fit(sample_train_data)
    assert diana.predict(sample_test_unique) == [0.0]


def test_predict_multiple(diana, sample_train_data, sample_test_data, sample_test_classes):
    diana.fit(sample_train_data)
    np.testing.assert_array_equal(
        diana.predict(sample_test_data), sample_test_classes
    )


def test_no_fit_predict(diana):
    with pytest.raises(NotFittedError):
        diana.predict([[2, 1, 3, 2]])


def test_not_correct_structure_predict(diana, sample_train_data):
    diana.fit(sample_train_data)
    # print(diana.n_features_in_)
    with pytest.raises(ValueError):
        diana.predict([[2, 1, 3]])
