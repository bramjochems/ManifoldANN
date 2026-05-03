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
    _vov_defined = False
    _to_matrix = None
    _to_vector = None
    _warmed_keys = set()
    _warmup_kind = None

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
            JuliaExternalWrapper._julia_initialized = True

        if not JuliaExternalWrapper._vov_defined:
            jl.seval("""
            function matrix_to_vector_of_vectors(M)
                [M[:, i] for i in 1:size(M, 2)]
            end
            """)
            JuliaExternalWrapper._vov_defined = True

        self._index = None
        self._data = None

    # ------------------------------------------------------------------
    # Lifecycle: marshal numpy -> Julia outside the timed region
    # ------------------------------------------------------------------

    def _matrix_form(self):
        """One of 'matrix' or 'vector_of_vectors'. Subclasses override."""
        return "matrix"

    def prepare_data(self, X):
        """Charge marshalling outside the timed `fit()` region."""
        X_fortran = np.asfortranarray(X.T, dtype=np.float32)
        matrix_jl = self._to_matrix(X_fortran)
        if self._matrix_form() == "vector_of_vectors":
            prepared = jl.matrix_to_vector_of_vectors(matrix_jl)
        else:
            prepared = matrix_jl

        # Per-dim JIT warmup keyed off (kind, dim).
        kind = type(self)._warmup_kind
        if kind is not None:
            key = (kind, int(X.shape[1]))
            if key not in JuliaExternalWrapper._warmed_keys:
                try:
                    self._warmup(int(X.shape[1]))
                except Exception as exc:
                    print(f"  ⚠ {kind} warmup at dim={X.shape[1]} failed: {exc}")
                JuliaExternalWrapper._warmed_keys.add(key)
        return prepared

    def prepare_queries(self, Q):
        Q_fortran = np.asfortranarray(Q.T, dtype=np.float32)
        return self._to_matrix(Q_fortran)

    def _warmup(self, dim: int) -> None:
        pass


class NearestNeighbors_KDTree(JuliaExternalWrapper):
    """Wrapper for NearestNeighbors.jl KDTree."""

    _warmup_kind = "NN_jl_KDTree"

    def __init__(self, metric, leafsize=10):
        """Initialize NearestNeighbors.jl KDTree.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
            leafsize: Leaf size for KD-Tree
        """
        super().__init__(metric)
        self.leafsize = leafsize

    def _warmup(self, dim: int) -> None:
        data = jl.seval(f"randn(Float32, {dim}, 128)")
        q = jl.seval(f"randn(Float32, {dim})")
        idx = jl.KDTree(data, jl.Euclidean(), leafsize=self.leafsize)
        jl.knn(idx, q, 5)

    def fit(self, X) -> None:
        """Build the KDTree index. `X` is the prepared Julia matrix."""
        self._data = X

        if self._metric == "angular":
            import warnings
            warnings.warn(
                "NearestNeighbors.jl KDTree with angular metric: "
                "normalizing data and using Euclidean distance",
                UserWarning
            )
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

    def query_batch(self, queries, n: int) -> List[List[int]]:
        """Batch query via `NearestNeighbors.knn(tree, matrix, k)`.

        `queries` is the Julia matrix returned by `prepare_queries`. NN.jl
        runs the batch internally; one Julia↔Python boundary crossing.
        """
        if self._metric == "angular":
            # prepare_queries didn't normalise; do it here. Overhead is in
            # the timed region but symmetric with what `query()` does.
            queries = jl.normalize_cols(queries)
        idxs_jl, _dists_jl = jl.knn(self._index, queries, n)
        # idxs_jl is a Julia Vector{Vector{Int64}}; iterate to Python.
        return [[int(i) - 1 for i in row] for row in idxs_jl]

    def __str__(self) -> str:
        return f"NearestNeighbors-KDTree(leafsize={self.leafsize})"

    @staticmethod
    def is_available() -> bool:
        try:
            from juliacall import Main as jl
            return True
        except ImportError:
            return False

    @staticmethod
    def get_name() -> str:
        return "NearestNeighbors-KDTree"


class HNSW_jl(JuliaExternalWrapper):
    """Wrapper for HNSW.jl."""

    _warmup_kind = "HNSW_jl"

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

    def _matrix_form(self):
        return "vector_of_vectors"

    def _warmup(self, dim: int) -> None:
        data_mat = jl.seval(f"randn(Float32, {dim}, 128)")
        data_vov = jl.matrix_to_vector_of_vectors(data_mat)
        q = jl.seval(f"randn(Float32, {dim})")
        metric_obj = jl.CosineDist() if self._metric == "angular" else jl.Euclidean()
        idx = jl.HierarchicalNSW(
            data_vov, metric=metric_obj, M=8, efConstruction=40, ef=16,
        )
        jl.seval("add_to_graph!")(idx)
        try:
            jl.knn_search(idx, q, 5)
        except Exception:
            pass

    def fit(self, X) -> None:
        """Build the HNSW index. `X` is the prepared Julia vector-of-vectors."""
        self._data = X

        if self._metric == "angular":
            metric_obj = jl.CosineDist()
        else:
            metric_obj = jl.Euclidean()

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

    def prepare_queries(self, Q):
        """HNSW.jl's batch knn_search wants a `Vector{<:AbstractVector}`."""
        Q_fortran = np.asfortranarray(Q.T, dtype=np.float32)
        matrix_jl = self._to_matrix(Q_fortran)
        return jl.matrix_to_vector_of_vectors(matrix_jl)

    def query_batch(self, queries, n: int) -> List[List[int]]:
        """Batch query via `HNSW.knn_search(hnsw, vec_of_vecs, K)`.

        `queries` is the Julia vector-of-vectors from `prepare_queries`.
        HNSW.jl threads the batch internally; one Julia↔Python crossing for
        the search call. Result is `Vector{Vector{UInt32}}` (one entry per
        query). We pre-cast through Int32 to widen for Python and let
        juliacall iterate the outer vector, converting each row in one go.
        """
        idxs_jl, _dists_jl = jl.knn_search(self._index, queries, n)
        # idxs_jl is Vector{Vector{UInt32}}. Iterate the outer vector once
        # in Python; each inner vector is converted to a numpy uint32 array
        # zero-copy by juliacall, then we cast to int and decrement.
        return [[int(i) - 1 for i in row] for row in idxs_jl]

    def __str__(self) -> str:
        return f"HNSW.jl(M={self.M}, ef_construction={self.ef_construction}, ef={self.ef})"

    @staticmethod
    def is_available() -> bool:
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

    _warmup_kind = "NND_jl"

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

    def _matrix_form(self):
        return "vector_of_vectors"

    def _warmup(self, dim: int) -> None:
        data_mat = jl.seval(f"randn(Float32, {dim}, 128)")
        data_vov = jl.matrix_to_vector_of_vectors(data_mat)
        q = jl.seval(f"randn(Float32, {dim})")
        metric_obj = jl.CosineDist() if self._metric == "angular" else jl.Euclidean()
        graph = jl.nndescent(
            data_vov, 8, metric_obj,
            max_iters=2, sample_rate=0.5, precision=0.1,
        )
        try:
            jl.search(graph, [q], 5, max_candidates=16)
        except Exception:
            pass

    def fit(self, X) -> None:
        """Build the NN-Descent graph. `X` is the prepared Julia vector-of-vectors."""
        self._data = X

        if self._metric == "angular":
            metric_obj = jl.CosineDist()
        else:
            metric_obj = jl.Euclidean()

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

        indices, distances = jl.search(self._graph, [query_jl], n, max_candidates=self.max_candidates)

        result_indices = indices[:, 0]

        return [int(idx) - 1 for idx in result_indices]

    def query_batch(self, queries, n: int) -> List[List[int]]:
        """Batch query via `NND.search(graph, matrix, k; max_candidates)`.

        `queries` is the Julia matrix from `prepare_queries`. NND.jl runs
        the batch internally; one Julia↔Python crossing. Result is a
        `Matrix{Int}` of shape (K, n_queries).
        """
        idxs_jl, _dists_jl = jl.search(
            self._graph, queries, n, max_candidates=self.max_candidates,
        )
        idxs_np = np.asarray(idxs_jl, dtype=np.int64)
        return (idxs_np - 1).T.tolist()

    def __str__(self) -> str:
        return (f"NearestNeighborDescent.jl(k={self.k}, max_iterations={self.max_iterations}, "
                f"sample_rate={self.sample_rate}, precision={self.precision}, max_candidates={self.max_candidates})")

    @staticmethod
    def is_available() -> bool:
        try:
            from juliacall import Main as jl
            return True
        except ImportError:
            return False

    @staticmethod
    def get_name() -> str:
        return "NearestNeighborDescent-jl"
