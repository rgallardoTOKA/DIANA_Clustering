import numpy as np
import pytest

from utils.calculate_similarity import SimilarityMeasure, cosine, l2_norm


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
