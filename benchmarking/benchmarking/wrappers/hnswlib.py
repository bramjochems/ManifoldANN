"""Wrapper for HNSWlib."""

from typing import List
import numpy as np

from .base import BaseANNWrapper


class HNSWlib(BaseANNWrapper):
    """Wrapper for HNSWlib to match our interface."""

    def __init__(self, metric, M=16, ef_construction=200, ef_search=50):
        """Initialize HNSWlib wrapper.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
            M: Maximum number of connections per element
            ef_construction: Size of dynamic candidate list during construction
            ef_search: Size of dynamic candidate list during search
        """
        super().__init__(metric)
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.index = None
        self.dimension = None
        self._num_threads = None

    def set_num_threads(self, n: int) -> None:
        """Store thread count; applied during fit/query."""
        self._num_threads = int(n)

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """Charge dtype/layout coercion outside the timed region."""
        return np.ascontiguousarray(X, dtype=np.float32)

    def fit(self, X: np.ndarray) -> None:
        """Build the HNSWlib index.

        Args:
            X: Training data, shape (n_samples, n_features)
        """
        import hnswlib

        self.dimension = X.shape[1]
        n_samples = X.shape[0]

        # Map metric name
        space_name = "cosine" if self._metric == "angular" else "l2"

        # Initialize index
        self.index = hnswlib.Index(space=space_name, dim=self.dimension)
        self.index.init_index(
            max_elements=n_samples, ef_construction=self.ef_construction, M=self.M
        )
        if self._num_threads is not None:
            self.index.set_num_threads(self._num_threads)

        # Add data
        self.index.add_items(X, np.arange(n_samples))

        # Set ef for queries
        self.index.set_ef(self.ef_search)
        if self._num_threads is not None:
            # hnswlib has separate build/query thread knobs; reset for query.
            self.index.set_num_threads(self._num_threads)

    def query(self, v: np.ndarray, n: int) -> List[int]:
        """Query for nearest neighbors.

        Args:
            v: Query vector, shape (n_features,)
            n: Number of neighbors to return

        Returns:
            List of neighbor indices
        """
        labels, distances = self.index.knn_query(v.reshape(1, -1), k=n)
        return labels[0].tolist()

    def query_batch(self, queries: np.ndarray, n: int) -> List[List[int]]:
        """Batch query — hnswlib parallelises internally across `set_num_threads`.

        Crosses the C++↔Python boundary once for the whole batch instead of
        n_queries individual `knn_query(reshape(1,-1))` round-trips. Matches
        what ManifoldANN's batch path does (one Julia↔Python crossing with
        internal parallelism) so the comparison is fair.
        """
        labels, _distances = self.index.knn_query(queries, k=n)
        return labels.tolist()

    def __str__(self) -> str:
        return (
            f"HNSWlib(M={self.M}, ef_construction={self.ef_construction}, "
            f"ef_search={self.ef_search})"
        )

    @staticmethod
    def is_available() -> bool:
        """Check if HNSWlib is installed."""
        try:
            import hnswlib
            return True
        except ImportError:
            return False

    @staticmethod
    def get_name() -> str:
        return "HNSWlib"
