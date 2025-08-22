# L2 and Cosine similarity measurement
from typing import Callable

import numpy as np
from numpy.typing import NDArray


def l2_norm(
    a: NDArray[np.float64], b: NDArray[np.float64]
) -> np.floating:  # function to calculate L2 norm of a vector
    """Calculates similarity with L2 norm

    Args:
        a (NDArray[np.float64]): Vector A
        b (NDArray[np.float64]): Vector B

    Returns:
        np.floating: L2 norm similiarity between vectors
    """
    x = a - b
    return np.linalg.norm(x)


def cosine(a: NDArray[np.float64], b: NDArray[np.float64]) -> np.floating:
    """Caclulates cosine similarity for two vectors

    Args:
        a (NDArray[np.float64]): Vector A
        b (NDArray[np.float64]): Vector B

    Returns:
        np.floating: Cosine similarity for vectors
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def SimilarityMeasure(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    func: Callable[[NDArray[np.float64], NDArray[np.float64]], np.floating] = l2_norm,
) -> np.floating:
    """Facade function to perform similarity meassure

    Args:
        a (NDArray[np.float64]): Vector A
        b (NDArray[np.float64]): Vector B
        func (Callable[[NDArray[np.float64], NDArray[np.float64]], np.floating], optional): Function to perform similarity by, accepts l2_norm and cosine. Defaults to l2_norm.

    Raises:
        ValueError: If func is not correct raise Error to inform user

    Returns:
        np.floating: Similarity between vectors
    """
    try:
        return func(a, b)
    except Exception:
        raise ValueError("Please provide valid similarity measurement type")
