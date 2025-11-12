"""Wrappers for external Julia ANN libraries (NearestNeighbors.jl, HNSW.jl)."""

import os
from typing import List
import numpy as np
from juliacall import Main as jl

from .base import BaseANNWrapper

# Path to Julia benchmark environment
JULIA_BENCHMARK_ENV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "julia")
)


class JuliaExternalWrapper(BaseANNWrapper):
    """Base wrapper for external Julia ANN libraries."""

    _julia_initialized = False
    _to_matrix = None
    _to_vector = None

    def __init__(self, metric):
        """Initialize wrapper and Julia environment.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
        """
        super().__init__(metric)

        # Initialize Julia benchmark environment (only once)
        if not JuliaExternalWrapper._julia_initialized:
            jl.seval(f'using Pkg; Pkg.activate("{JULIA_BENCHMARK_ENV}")')
            # Load packages
            jl.seval("using NearestNeighbors, HNSW, NearestNeighborDescent, Distances")

            # Create conversion functions
            JuliaExternalWrapper._to_matrix = jl.seval("x -> Matrix{Float32}(x)")
            JuliaExternalWrapper._to_vector = jl.seval("x -> Vector{Float32}(x)")

            # JIT warmup: compile functions with small dummy data to exclude compilation
            # time from benchmarks (same as ManifoldANN does)
            print("Warming up Julia external libraries (first use only)...")

            try:
                # Create warmup data: 10 dimensions, 100 points (column-major: each column is a point)
                warmup_data = jl.seval("randn(Float32, 10, 100)")
                warmup_query = jl.seval("randn(Float32, 10)")

                # Warmup NearestNeighbors.jl
                kdtree_warmup = jl.KDTree(warmup_data, jl.Euclidean())
                jl.knn(kdtree_warmup, warmup_query, 5)
                print("  ✓ NearestNeighbors.jl warmed up")
            except Exception as e:
                print(f"  ⚠ NearestNeighbors.jl warmup failed: {e}")

            # Note: HNSW.jl has API issues with querying, so we skip its warmup
            # The library will be disabled in benchmarks

            print("✓ Julia warmup complete")
            JuliaExternalWrapper._julia_initialized = True

        self._index = None
        self._data = None

    def _prepare_data(self, X: np.ndarray):
        """Convert numpy array to Julia format.

        Args:
            X: numpy array of shape (n_samples, n_features)
        """
        # Julia expects (n_features, n_samples)
        X_fortran = np.asfortranarray(X.T, dtype=np.float32)
        self._data = self._to_matrix(X_fortran)


class NearestNeighbors_KDTree(JuliaExternalWrapper):
    """Wrapper for NearestNeighbors.jl KDTree."""

    def __init__(self, metric, leafsize=10):
        """Initialize NearestNeighbors.jl KDTree.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
            leafsize: Leaf size for KD-Tree
        """
        super().__init__(metric)
        self.leafsize = leafsize

    def fit(self, X: np.ndarray) -> None:
        """Build the KDTree index."""
        self._prepare_data(X)

        # Map metric
        if self._metric == "angular":
            # Use cosine distance (1 - dot product for normalized vectors)
            # Note: NearestNeighbors.jl doesn't have built-in cosine, so normalize first
            import warnings
            warnings.warn(
                "NearestNeighbors.jl KDTree with angular metric: "
                "normalizing data and using Euclidean distance",
                UserWarning
            )
            # Normalize in Julia
            jl.seval("""
            function normalize_cols(X)
                X_norm = similar(X)
                for i in 1:size(X, 2)
                    col = X[:, i]
                    X_norm[:, i] = col / norm(col)
                end
                X_norm
            end
            """)
            self._data = jl.normalize_cols(self._data)
            metric_obj = jl.Euclidean()
        else:
            metric_obj = jl.Euclidean()

        # Build KDTree
        self._index = jl.KDTree(self._data, metric_obj, leafsize=self.leafsize)

    def query(self, v: np.ndarray, n: int) -> List[int]:
        """Query for nearest neighbors."""
        query_vec = np.asfortranarray(v, dtype=np.float32)

        # Normalize if using angular metric
        if self._metric == "angular":
            query_vec = query_vec / np.linalg.norm(query_vec)

        query_jl = self._to_vector(query_vec)

        # knn returns (indices, distances)
        indices, distances = jl.knn(self._index, query_jl, n)

        # Convert from Julia 1-indexed to Python 0-indexed
        return [int(idx) - 1 for idx in indices]

    def __str__(self) -> str:
        return f"NearestNeighbors-KDTree(leafsize={self.leafsize})"

    @staticmethod
    def is_available() -> bool:
        """Check if NearestNeighbors.jl is available."""
        try:
            from juliacall import Main as jl
            # Check if we can activate the Julia environment
            return True
        except ImportError:
            return False

    @staticmethod
    def get_name() -> str:
        return "NearestNeighbors-KDTree"


class HNSW_jl(JuliaExternalWrapper):
    """Wrapper for HNSW.jl."""

    _converter_defined = False

    def __init__(self, metric, M=16, ef_construction=200, ef=50):
        """Initialize HNSW.jl.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
            M: Maximum number of connections per element
            ef_construction: Size of dynamic candidate list during construction
            ef: Size of dynamic candidate list during search
        """
        super().__init__(metric)
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef

        # Define conversion function once (outside of fit timing)
        if not HNSW_jl._converter_defined:
            jl.seval("""
            function matrix_to_vector_of_vectors(M)
                [M[:, i] for i in 1:size(M, 2)]
            end
            """)
            HNSW_jl._converter_defined = True

    def fit(self, X: np.ndarray) -> None:
        """Build the HNSW index."""
        # HNSW.jl requires data as a vector of vectors, not a matrix
        # Convert: (n_samples, n_features) -> Vector{Vector{Float32}}
        X_fortran = np.asfortranarray(X.T, dtype=np.float32)  # (n_features, n_samples)

        # Convert to Julia vector of vectors using pre-defined function
        matrix_jl = self._to_matrix(X_fortran)
        self._data = jl.matrix_to_vector_of_vectors(matrix_jl)

        # Map metric
        if self._metric == "angular":
            # HNSW.jl uses Distances.jl metrics
            metric_obj = jl.CosineDist()
        else:
            metric_obj = jl.Euclidean()

        # Build HNSW index
        self._index = jl.HierarchicalNSW(
            self._data,
            metric=metric_obj,
            M=self.M,
            efConstruction=self.ef_construction,
            ef=self.ef
        )

        # Add all points to the graph (required for HNSW.jl)
        jl.seval("add_to_graph!")(self._index)

    def query(self, v: np.ndarray, n: int) -> List[int]:
        """Query for nearest neighbors."""
        query_vec = np.asfortranarray(v, dtype=np.float32)
        query_jl = self._to_vector(query_vec)

        # Search
        indices, distances = jl.knn_search(self._index, query_jl, n)

        # Convert from Julia 1-indexed to Python 0-indexed
        return [int(idx) - 1 for idx in indices]

    def __str__(self) -> str:
        return f"HNSW.jl(M={self.M}, ef_construction={self.ef_construction}, ef={self.ef})"

    @staticmethod
    def is_available() -> bool:
        """Check if HNSW.jl is available."""
        try:
            from juliacall import Main as jl
            return True
        except ImportError:
            return False

    @staticmethod
    def get_name() -> str:
        return "HNSW-jl"


class NearestNeighborDescent_jl(JuliaExternalWrapper):
    """Wrapper for NearestNeighborDescent.jl."""

    _converter_defined = False

    def __init__(self, metric, k=32, max_iterations=10, sample_rate=1.0, precision=0.001, max_candidates=None):
        """Initialize NearestNeighborDescent.jl.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
            k: Number of nearest neighbors to find (graph construction)
            max_iterations: Maximum number of iterations
            sample_rate: Fraction of neighbors to sample (0.0 to 1.0)
            precision: Convergence threshold
            max_candidates: Search beam width (defaults to max(k, 20))
        """
        super().__init__(metric)
        self.k = k
        self.max_iterations = max_iterations
        self.sample_rate = sample_rate
        self.precision = precision
        self.max_candidates = max_candidates if max_candidates is not None else max(k, 20)

        # Define conversion function once (outside of fit timing)
        if not NearestNeighborDescent_jl._converter_defined:
            jl.seval("""
            function matrix_to_vector_of_vectors(M)
                [M[:, i] for i in 1:size(M, 2)]
            end
            """)
            NearestNeighborDescent_jl._converter_defined = True

    def fit(self, X: np.ndarray) -> None:
        """Build the NN-Descent graph."""
        # NearestNeighborDescent.jl accepts both matrices and vector of vectors
        # Using vector of vectors for consistency with HNSW.jl
        X_fortran = np.asfortranarray(X.T, dtype=np.float32)  # (n_features, n_samples)

        # Convert to Julia vector of vectors
        matrix_jl = self._to_matrix(X_fortran)
        self._data = jl.matrix_to_vector_of_vectors(matrix_jl)

        # Map metric
        if self._metric == "angular":
            metric_obj = jl.CosineDist()
        else:
            metric_obj = jl.Euclidean()

        # Build NN-Descent graph
        self._graph = jl.nndescent(
            self._data,
            self.k,
            metric_obj,
            max_iters=self.max_iterations,
            sample_rate=self.sample_rate,
            precision=self.precision
        )

    def query(self, v: np.ndarray, n: int) -> List[int]:
        """Query for nearest neighbors."""
        query_vec = np.asfortranarray(v, dtype=np.float32)
        query_jl = self._to_vector(query_vec)

        # Search using the graph with max_candidates parameter
        # search returns (indices, distances) as matrices where:
        #   - rows = neighbors (k)
        #   - columns = queries (1 in our case)
        indices, distances = jl.search(self._graph, [query_jl], n, max_candidates=self.max_candidates)

        # Extract the first (and only) column: indices[:, 1] (Julia 1-indexed)
        # But from Python with juliacall, we use Python indexing: [:, 0]
        result_indices = indices[:, 0]

        # Convert from Julia 1-indexed to Python 0-indexed
        return [int(idx) - 1 for idx in result_indices]

    def __str__(self) -> str:
        return (f"NearestNeighborDescent.jl(k={self.k}, max_iterations={self.max_iterations}, "
                f"sample_rate={self.sample_rate}, precision={self.precision}, max_candidates={self.max_candidates})")

    @staticmethod
    def is_available() -> bool:
        """Check if NearestNeighborDescent.jl is available."""
        try:
            from juliacall import Main as jl
            return True
        except ImportError:
            return False

    @staticmethod
    def get_name() -> str:
        return "NearestNeighborDescent-jl"
