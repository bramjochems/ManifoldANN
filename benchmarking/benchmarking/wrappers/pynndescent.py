"""Wrapper for PyNNDescent (Python NN-Descent implementation)."""

from typing import List
import numpy as np

from .base import BaseANNWrapper


class PyNNDescent(BaseANNWrapper):
    """Wrapper for PyNNDescent to match our interface."""

    def __init__(self, metric, n_neighbors=30, diversify_prob=1.0, pruning_degree_multiplier=1.5):
        """Initialize PyNNDescent wrapper.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
            n_neighbors: Number of neighbors for graph construction
            diversify_prob: Probability of using diversification (1.0 = always)
            pruning_degree_multiplier: Degree multiplier for pruning
        """
        super().__init__(metric)
        self.n_neighbors = n_neighbors
        self.diversify_prob = diversify_prob
        self.pruning_degree_multiplier = pruning_degree_multiplier
        self.index = None
        self._num_threads = None

    def set_num_threads(self, n: int) -> None:
        """PyNNDescent honours `n_jobs` (numba-backed parallelism)."""
        self._num_threads = int(n)

    def fit(self, X: np.ndarray) -> None:
        """Build the PyNNDescent index.

        Args:
            X: Training data, shape (n_samples, n_features)
        """
        from pynndescent import NNDescent

        # Map metric name
        metric_name = "cosine" if self._metric == "angular" else "euclidean"

        # Build index
        kwargs = dict(
            metric=metric_name,
            n_neighbors=self.n_neighbors,
            diversify_prob=self.diversify_prob,
            pruning_degree_multiplier=self.pruning_degree_multiplier,
        )
        if self._num_threads is not None:
            kwargs["n_jobs"] = self._num_threads
        self.index = NNDescent(X, **kwargs)

    def query(self, v: np.ndarray, n: int) -> List[int]:
        """Query for nearest neighbors.

        Args:
            v: Query vector, shape (n_features,)
            n: Number of neighbors to return

        Returns:
            List of neighbor indices
        """
        indices, distances = self.index.query(v.reshape(1, -1), k=n)
        return indices[0].tolist()

    def query_batch(self, queries: np.ndarray, n: int) -> List[List[int]]:
        """Batch query — PyNNDescent's `query` accepts a `(n_queries, d)`
        matrix natively and parallelises across `n_jobs` (numba)."""
        indices, _distances = self.index.query(queries, k=n)
        return indices.tolist()

    def __str__(self) -> str:
        return (
            f"PyNNDescent(n_neighbors={self.n_neighbors}, "
            f"diversify_prob={self.diversify_prob}, "
            f"pruning_mult={self.pruning_degree_multiplier})"
        )

    @staticmethod
    def is_available() -> bool:
        """Check if PyNNDescent is installed."""
        try:
            import pynndescent
            return True
        except ImportError:
            return False

    @staticmethod
    def get_name() -> str:
        return "PyNNDescent"
