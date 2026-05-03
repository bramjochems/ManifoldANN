"""Base wrapper interface for ANN algorithms."""

from abc import ABC, abstractmethod
from typing import Any, List
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

    # ------------------------------------------------------------------
    # Lifecycle hooks for fair timing
    # ------------------------------------------------------------------

    def prepare_data(self, X: np.ndarray) -> Any:
        """Prepare training data outside the timed build region.

        Wrappers that need to perform data conversions (numpy -> Julia
        matrix, dtype/layout coercion, normalisation) should do that here so
        the cost is *not* charged to the timed `fit()` region.

        The harness calls this once outside the timed region and passes the
        returned object to `fit`.

        Default: pass-through.
        """
        return X

    def prepare_queries(self, Q: np.ndarray) -> Any:
        """Prepare a batch of query points outside the timed query region.

        Same rationale as `prepare_data` but for queries. Default
        pass-through. Override only if your wrapper wants to amortise
        per-batch conversion cost outside the timed region.
        """
        return Q

    def set_num_threads(self, n: int) -> None:
        """Set library-specific build/query thread count.

        Default: no-op. Wrappers backing libraries with their own thread
        knobs (hnswlib's `set_num_threads`, FAISS's `omp_set_num_threads`,
        pynndescent's `n_jobs`, scipy's per-query `n_jobs`) should override
        and store / forward `n` so all libraries run with matched
        parallelism. Wrappers that piggy-back on Julia's thread pool
        (everything in `manifoldann.py` / `julia_external.py`) can leave
        this as a no-op since `JULIA_NUM_THREADS` is set process-wide.
        """
        return None

    @abstractmethod
    def fit(self, X) -> None:
        """Build the index from training data.

        Args:
            X: Training data (output of `prepare_data`). For wrappers that
               do not override `prepare_data`, this is a numpy array of
               shape (n_samples, n_features).
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
