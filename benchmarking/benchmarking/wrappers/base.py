"""Base wrapper interface for ANN algorithms."""

from abc import ABC, abstractmethod
from typing import List
import numpy as np


class BaseANNWrapper(ABC):
    """Abstract base class for ANN algorithm wrappers.

    All algorithm wrappers should inherit from this class and implement
    the required methods.
    """

    def __init__(self, metric: str):
        """Initialize the wrapper.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
        """
        if metric not in ["angular", "euclidean"]:
            raise ValueError(f"Invalid metric: {metric}. Must be 'angular' or 'euclidean'")
        self._metric = metric

    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        """Build the index from training data.

        Args:
            X: Training data, shape (n_samples, n_features)
        """
        pass

    @abstractmethod
    def query(self, v: np.ndarray, n: int) -> List[int]:
        """Query for nearest neighbors.

        Args:
            v: Query vector, shape (n_features,)
            n: Number of neighbors to return

        Returns:
            List of neighbor indices
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return string representation of the algorithm configuration."""
        pass

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Check if the required library is installed.

        Returns:
            True if library is available, False otherwise
        """
        pass

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        """Get the canonical name of this algorithm.

        Returns:
            Algorithm name (e.g., 'Annoy', 'ManifoldANN-HNSW')
        """
        pass
