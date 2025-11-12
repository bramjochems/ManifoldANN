"""Wrapper for Annoy (Approximate Nearest Neighbors Oh Yeah)."""

from typing import List
import numpy as np

from .base import BaseANNWrapper


class Annoy(BaseANNWrapper):
    """Wrapper for Annoy to match our interface."""

    def __init__(self, metric, n_trees=10):
        """Initialize Annoy wrapper.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
            n_trees: Number of trees to build (more trees = better recall, slower build)
        """
        super().__init__(metric)
        self.n_trees = n_trees
        self.index = None
        self.dimension = None

    def fit(self, X: np.ndarray) -> None:
        """Build the Annoy index.

        Args:
            X: Training data, shape (n_samples, n_features)
        """
        from annoy import AnnoyIndex

        self.dimension = X.shape[1]
        metric_name = "angular" if self._metric == "angular" else "euclidean"
        self.index = AnnoyIndex(self.dimension, metric_name)

        for i, vec in enumerate(X):
            self.index.add_item(i, vec.tolist())

        self.index.build(self.n_trees)

    def query(self, v: np.ndarray, n: int) -> List[int]:
        """Query for nearest neighbors.

        Args:
            v: Query vector, shape (n_features,)
            n: Number of neighbors to return

        Returns:
            List of neighbor indices
        """
        return self.index.get_nns_by_vector(v.tolist(), n)

    def __str__(self) -> str:
        return f"Annoy(n_trees={self.n_trees})"

    @staticmethod
    def is_available() -> bool:
        """Check if Annoy is installed."""
        try:
            import annoy
            return True
        except ImportError:
            return False

    @staticmethod
    def get_name() -> str:
        return "Annoy"
