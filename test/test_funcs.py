import numpy as np
import pandas as pd
import pytest

from src.DIvisiveANAlysis import DianaClustering
from utils.calculate_similarity import SimilarityMeasure, cosine, l2_norm


# / Testing Utils
@pytest.fixture
def sample_data():
    return np.array([2, 1, 2, 3, 2, 9]), np.array([3, 4, 2, 4, 5, 5])


def test_l2(sample_data):
    a, b = sample_data
    assert l2_norm(a, b) == 6.0


def test_cosine(sample_data):
    a, b = sample_data
    assert cosine(a, b) == pytest.approx(0.8188504723485274)


def test_Similarity_l2(sample_data):
    a, b = sample_data
    assert SimilarityMeasure(a=a, b=b, func=l2_norm) == 6.0


def test_Similarity_cosine(sample_data):
    a, b = sample_data
    assert SimilarityMeasure(a=a, b=b, func=cosine) == pytest.approx(0.8188504723485274)


# / Testing DIANA
@pytest.fixture
def diana():
    return DianaClustering(data=pd.read_csv("test/data/HAYES_ROTH.csv"))


@pytest.fixture
def sample_classes():
    return np.load("test/data/example_classes.npy")


def test_sim_matrix(diana):
    assert diana.similarity_matrix.shape == (132, 132)


# TODO: is this not deterministic?
def test_fit(diana, sample_classes):
    np.testing.assert_array_equal(diana.fit(3), sample_classes)
