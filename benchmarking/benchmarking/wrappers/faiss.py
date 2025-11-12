"""Wrapper for FAISS (Facebook AI Similarity Search)."""

from typing import List
import numpy as np

from .base import BaseANNWrapper


class FAISS_IVF(BaseANNWrapper):
    """Wrapper for FAISS IVF (Inverted File) index."""

    def __init__(self, metric, nlist=100, nprobe=10):
        """Initialize FAISS IVF wrapper.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
            nlist: Number of clusters for IVF
            nprobe: Number of clusters to visit during search
        """
        super().__init__(metric)
        self.nlist = nlist
        self.nprobe = nprobe
        self.index = None
        self.dimension = None

    def fit(self, X: np.ndarray) -> None:
        """Build the FAISS IVF index.

        Args:
            X: Training data, shape (n_samples, n_features)
        """
        import faiss

        self.dimension = X.shape[1]
        X = X.astype(np.float32)

        # Normalize vectors for angular distance
        if self._metric == "angular":
            faiss.normalize_L2(X)

        # Create quantizer and IVF index
        quantizer = faiss.IndexFlatL2(self.dimension)

        if self._metric == "angular":
            # For angular, use inner product after normalization
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
        else:
            # For Euclidean, use L2
            self.index = faiss.IndexIVFFlat(
                quantizer, self.dimension, self.nlist, faiss.METRIC_L2
            )

        # Train and add data
        self.index.train(X)
        self.index.add(X)

        # Set nprobe for queries
        self.index.nprobe = self.nprobe

    def query(self, v: np.ndarray, n: int) -> List[int]:
        """Query for nearest neighbors.

        Args:
            v: Query vector, shape (n_features,)
            n: Number of neighbors to return

        Returns:
            List of neighbor indices
        """
        import faiss

        v = v.astype(np.float32).reshape(1, -1)

        # Normalize for angular distance
        if self._metric == "angular":
            faiss.normalize_L2(v)

        distances, indices = self.index.search(v, n)
        return indices[0].tolist()

    def __str__(self) -> str:
        return f"FAISS-IVF(nlist={self.nlist}, nprobe={self.nprobe})"

    @staticmethod
    def is_available() -> bool:
        """Check if FAISS is installed."""
        try:
            import faiss
            return True
        except ImportError:
            return False

    @staticmethod
    def get_name() -> str:
        return "FAISS-IVF"
