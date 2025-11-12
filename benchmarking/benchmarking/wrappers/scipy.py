"""Wrapper for SciPy's KDTree."""

from typing import List
import numpy as np

from .base import BaseANNWrapper


class SciPy_KDTree(BaseANNWrapper):
    """Wrapper for SciPy's cKDTree (exact nearest neighbors for comparison)."""

    def __init__(self, metric, leafsize=32):
        """Initialize SciPy KDTree wrapper.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
            leafsize: Number of points at which to switch to brute-force
        """
        super().__init__(metric)

        if metric == "angular":
            import warnings
            warnings.warn(
                "SciPy KDTree only supports Euclidean distance. "
                "Angular metric not supported - algorithm will be skipped.",
                UserWarning
            )

        self.leafsize = leafsize
        self.index = None

    def fit(self, X: np.ndarray) -> None:
        """Build the KDTree index.

        Args:
            X: Training data, shape (n_samples, n_features)
        """
        if self._metric != "euclidean":
            raise ValueError("SciPy KDTree only supports Euclidean distance")

        from scipy.spatial import cKDTree

        self.index = cKDTree(X, leafsize=self.leafsize)

    def query(self, v: np.ndarray, n: int) -> List[int]:
        """Query for nearest neighbors.

        Args:
            v: Query vector, shape (n_features,)
            n: Number of neighbors to return

        Returns:
            List of neighbor indices
        """
        distances, indices = self.index.query(v, k=n)

        # Handle single neighbor case
        if n == 1:
            return [int(indices)]
        return indices.tolist()

    def __str__(self) -> str:
        return f"SciPy-KDTree(leafsize={self.leafsize})"

    @staticmethod
    def is_available() -> bool:
        """Check if SciPy is installed."""
        try:
            from scipy.spatial import cKDTree
            return True
        except ImportError:
            return False

    @staticmethod
    def get_name() -> str:
        return "SciPy-KDTree"
