"""Wrappers for ManifoldANN (Julia) algorithms."""

import os
import numpy as np
from juliacall import Main as jl

from .base import BaseANNWrapper

# Configure Julia environment to use the ManifoldANN project
MANIFOLDANN_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


class ManifoldANNWrapper(BaseANNWrapper):
    """Base wrapper class for ManifoldANN (Julia) algorithms."""

    # Class-level converters (created once)
    _julia_initialized = False
    _to_matrix = None
    _to_vector = None

    def __init__(self, metric):
        """Initialize the wrapper.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
        """
        super().__init__(metric)

        # Initialize Julia and load ManifoldANN (only once per class)
        if not ManifoldANNWrapper._julia_initialized:
            jl.seval(f'using Pkg; Pkg.activate("{MANIFOLDANN_PATH}")')
            jl.seval("using ManifoldANN")
            # Create conversion functions once
            ManifoldANNWrapper._to_matrix = jl.seval("x -> Matrix{Float32}(x)")
            ManifoldANNWrapper._to_vector = jl.seval("x -> Vector{Float32}(x)")

            # JIT warmup: compile core functions with small dummy data
            # This ensures compilation time is excluded from benchmark timings
            print("Warming up ManifoldANN (first use only)...")
            warmup_data = jl.seval("randn(Float32, 10, 100)")  # 10-dim, 100 points
            warmup_query = jl.seval("randn(Float32, 10)")

            # Warmup a simple index (BruteForce is fastest to compile)
            warmup_index = jl.build_index(jl.BruteForceIndex, warmup_data)
            jl.query(warmup_index, warmup_data, warmup_query, 5)

            print("✓ ManifoldANN warmup complete")
            ManifoldANNWrapper._julia_initialized = True

        self._index = None
        self._data = None

    def _get_distance_function(self):
        """Get the appropriate Julia distance function for the metric.

        Returns:
            Julia function for computing distances (squared variant for use in priority queues)
        """
        if self._metric == "angular":
            return jl.ManifoldANN.squared_cosine_distance
        else:  # euclidean
            return jl.ManifoldANN.default_squared_distance

    def fit(self, X):
        """Build the index from training data.

        Args:
            X: numpy array of shape (n_samples, n_features)
        """
        # Convert to Fortran-contiguous array (column-major) for Julia
        # Julia expects (n_features, n_samples) while numpy gives (n_samples, n_features)
        X_fortran = np.asfortranarray(X.T, dtype=np.float32)

        # Convert to Julia array using pre-created converter
        self._data = self._to_matrix(X_fortran)

    def query(self, v, n):
        """Query for nearest neighbors.

        Args:
            v: Query vector (1D numpy array)
            n: Number of neighbors to return

        Returns:
            List of neighbor indices (0-indexed for Python)
        """
        # Convert to Julia Vector{Float32} using pre-created converter
        query_vec = np.asfortranarray(v, dtype=np.float32)
        query_jl = self._to_vector(query_vec)

        # Call Julia query function
        # Note: Julia uses 1-based indexing, so we need to convert
        result = jl.query(self._index, self._data, query_jl, n)

        # Convert from Julia 1-indexed to Python 0-indexed
        return [int(idx) - 1 for idx in result]

    def query_batch(self, queries, n):
        """Query for nearest neighbors of multiple queries at once.

        This is significantly faster than calling query() in a loop because it
        minimizes Python↔Julia boundary crossings (1 crossing instead of N).

        Args:
            queries: numpy array of shape (n_queries, n_features)
            n: Number of neighbors to return per query

        Returns:
            List of lists: [[neighbor_indices for query1], [for query2], ...]
            All indices are 0-indexed for Python.
        """
        # Convert to Fortran-contiguous (column-major) for Julia
        # Julia expects (n_features, n_queries) while numpy gives (n_queries, n_features)
        queries_fortran = np.asfortranarray(queries.T, dtype=np.float32)
        queries_jl = self._to_matrix(queries_fortran)

        # Call Julia batch query function - crosses boundary only ONCE
        results_jl = jl.query(self._index, self._data, queries_jl, n)

        # Convert from Julia 1-indexed to Python 0-indexed
        return [[int(idx) - 1 for idx in result] for result in results_jl]

    @staticmethod
    def is_available():
        """Check if ManifoldANN is available."""
        try:
            from juliacall import Main as jl
            return True
        except ImportError:
            return False


class ManifoldANN_BruteForce(ManifoldANNWrapper):
    """Wrapper for ManifoldANN BruteForceIndex (baseline)."""

    def __init__(self, metric):
        """Initialize brute force wrapper."""
        super().__init__(metric)

    def fit(self, X):
        """Build the brute force index."""
        super().fit(X)
        distance_fn = self._get_distance_function()
        self._index = jl.build_index(jl.BruteForceIndex, self._data, distance=distance_fn)

    def __str__(self):
        return "ManifoldANN-BruteForce()"

    @staticmethod
    def get_name():
        return "ManifoldANN-BruteForce"


class ManifoldANN_LSH(ManifoldANNWrapper):
    """Wrapper for ManifoldANN LSHIndex."""

    def __init__(self, metric, n_tables=8, hash_length=16, bin_width=None):
        """Initialize LSH wrapper.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
            n_tables: Number of hash tables
            hash_length: Length of hash codes
            bin_width: Bin width for BinningHash (Euclidean LSH). If None and metric
                      is 'euclidean', will be auto-computed as 3 × avg_nn_distance
        """
        super().__init__(metric)
        self._n_tables = n_tables
        self._hash_length = hash_length
        self._bin_width = bin_width

    def fit(self, X):
        """Build the LSH index."""
        super().fit(X)

        # Select appropriate hash family based on metric
        if self._metric == "euclidean":
            # Use BinningHash (p-stable LSH) for Euclidean distance
            if self._bin_width is None:
                # Auto-compute bin_width using heuristic: w ≈ 3 × avg_nn_distance
                # Sample a subset of points to estimate average NN distance
                n_samples = min(1000, X.shape[0])
                indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
                sample = X[indices]

                # Compute pairwise distances and find nearest neighbor for each sample
                from scipy.spatial.distance import cdist

                distances = cdist(sample, sample, metric="euclidean")
                np.fill_diagonal(distances, np.inf)  # Ignore self-distances
                avg_nn_dist = np.mean(np.min(distances, axis=1))

                self._bin_width = 3.0 * avg_nn_dist
                print(
                    f"LSH: Auto-computed bin_width = {self._bin_width:.4f} "
                    f"(3 × avg_nn_dist = 3 × {avg_nn_dist:.4f})"
                )

            # Build index with BinningHash
            self._index = jl.build_index(
                jl.LSHIndex,
                self._data,
                n_tables=self._n_tables,
                hash_length=self._hash_length,
                hash_factory=jl.make_binning_hash,
                bin_width=self._bin_width,
                use_offset=True,
                T=jl.Float32,
            )
        else:
            # Use RandomHyperplaneHash for angular/cosine distance
            self._index = jl.build_index(
                jl.LSHIndex,
                self._data,
                n_tables=self._n_tables,
                hash_length=self._hash_length,
                hash_factory=jl.make_random_hyperplane_hash,
                T=jl.Float32,
            )

    def query(self, v, n):
        """Query for nearest neighbors with LSH."""
        query_vec = np.asfortranarray(v, dtype=np.float32)
        query_jl = self._to_vector(query_vec)

        # Call with candidate_cap if set
        if hasattr(self, "_candidate_cap") and self._candidate_cap is not None:
            result = jl.query(
                self._index, self._data, query_jl, n, candidate_cap=self._candidate_cap
            )
        else:
            result = jl.query(self._index, self._data, query_jl, n)

        return [int(idx) - 1 for idx in result]

    def __str__(self):
        cap_str = (
            f", candidate_cap={self._candidate_cap}"
            if hasattr(self, "_candidate_cap")
            else ""
        )
        bw_str = f", bin_width={self._bin_width:.4f}" if self._bin_width is not None else ""
        return f"ManifoldANN-LSH(n_tables={self._n_tables}, hash_length={self._hash_length}{bw_str}{cap_str})"

    @staticmethod
    def get_name():
        return "ManifoldANN-LSH"


class ManifoldANN_KDTree(ManifoldANNWrapper):
    """Wrapper for ManifoldANN KDTreeIndex."""

    _VALID_AXIS_SELECTORS = ("variance", "cyclic")

    def __init__(self, metric, axis_selector="variance"):
        """Initialize KDTree wrapper.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
            axis_selector: Strategy for choosing split axes (variance or cyclic)
        """
        super().__init__(metric)
        if axis_selector not in self._VALID_AXIS_SELECTORS:
            raise ValueError(
                f"axis_selector must be one of {self._VALID_AXIS_SELECTORS}, got '{axis_selector}'"
            )
        self._axis_selector = axis_selector

        if metric != "euclidean":
            import warnings

            warnings.warn(
                "KDTreeIndex currently tunes splits for Euclidean metrics; "
                "angular queries fall back to conservative pruning.",
                UserWarning,
            )

    def fit(self, X):
        """Build the KD-tree index."""
        super().fit(X)

        axis_symbol = jl.Symbol(self._axis_selector)
        self._index = jl.build_index(
            jl.KDTreeIndex,
            self._data,
            axis_selector=axis_symbol,
        )

    def query_batch(self, queries, n):
        """KDTreeIndex exposes only scalar queries; batch in Python."""
        return [self.query(q, n) for q in queries]

    def __str__(self):
        return f"ManifoldANN-KDTree(axis_selector={self._axis_selector})"

    @staticmethod
    def get_name():
        return "ManifoldANN-KDTree"


class ManifoldANN_HNSW(ManifoldANNWrapper):
    """Wrapper for ManifoldANN HNSWIndex."""

    _VALID_NEIGHBOR_POLICIES = {"heuristic", "diversified"}

    def __init__(
        self,
        metric,
        M=16,
        ef_construction=200,
        ef_search=64,
        neighbor_policy="diversified",
    ):
        """Initialize HNSW wrapper.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
            M: Maximum number of connections per element
            ef_construction: Size of dynamic candidate list during construction
            ef_search: Size of dynamic candidate list during search
            neighbor_policy: Strategy for neighbor selection ('heuristic' or 'diversified')
        """
        super().__init__(metric)
        self._M = M
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        if neighbor_policy not in self._VALID_NEIGHBOR_POLICIES:
            raise ValueError(
                f"neighbor_policy must be one of {self._VALID_NEIGHBOR_POLICIES}"
            )
        self._neighbor_policy = neighbor_policy

    def fit(self, X):
        """Build the HNSW index."""
        super().fit(X)

        # Build index using Julia with appropriate distance function
        neighbor_policy_symbol = jl.Symbol(self._neighbor_policy)
        distance_fn = self._get_distance_function()
        self._index = jl.build_index(
            jl.HNSWIndex,
            self._data,
            M=self._M,
            ef_construction=self._ef_construction,
            ef_search=self._ef_search,
            neighbor_policy=neighbor_policy_symbol,
            distance=distance_fn,
        )

    def query(self, v, n):
        """Query for nearest neighbors with HNSW."""
        query_vec = np.asfortranarray(v, dtype=np.float32)
        query_jl = self._to_vector(query_vec)

        # Call with ef_search parameter
        result = jl.query(self._index, self._data, query_jl, n, ef_search=self._ef_search)

        return [int(idx) - 1 for idx in result]

    def __str__(self):
        return (
            f"ManifoldANN-HNSW(M={self._M}, ef_construction={self._ef_construction}, "
            f"ef_search={self._ef_search}, neighbor_policy={self._neighbor_policy})"
        )

    @staticmethod
    def get_name():
        return "ManifoldANN-HNSW"


class ManifoldANN_NNDescent(ManifoldANNWrapper):
    """Wrapper for ManifoldANN NNDescentIndex."""

    def __init__(
        self,
        metric,
        k=32,
        max_iterations=5,
        convergence_threshold=0.01,
        sample_rate=0.5,
        symmetry_policy="pruned",
        ef_search=None,
    ):
        """Initialize NN-Descent wrapper.

        Args:
            metric: Distance metric ('angular' or 'euclidean')
            k: Number of neighbors per node in the graph (default: 32)
            max_iterations: Maximum NN-Descent iterations (default: 5, typically converges in 3-5)
            convergence_threshold: Relative improvement threshold to stop early (default: 0.01 = 1%)
            sample_rate: Fraction of candidate pairs to evaluate (default: 0.5 = 2x faster)
            symmetry_policy: Graph symmetry strategy (default: 'pruned' for 1.5x degree multiplier)
            ef_search: Beam width for graph search queries (default: 2*k)
        """
        super().__init__(metric)
        self._k = int(k)
        self._max_iterations = int(max_iterations)
        self._convergence_threshold = float(convergence_threshold)
        self._sample_rate = float(sample_rate)
        self._symmetry_policy = symmetry_policy
        self._ef_search = ef_search if ef_search is not None else max(self._k, 2 * self._k)

    def fit(self, X):
        """Build the NN-Descent index."""
        super().fit(X)
        sampling_policy = jl.ManifoldANN.UniformPairSampling(self._sample_rate)

        # Convert symmetry policy string to Julia symbol
        symmetry_symbol = jl.Symbol(self._symmetry_policy)
        distance_fn = self._get_distance_function()

        self._index = jl.build_index(
            jl.NNDescentIndex,
            self._data,
            k=self._k,
            max_iterations=self._max_iterations,
            convergence_threshold=self._convergence_threshold,
            sampling_policy=sampling_policy,
            symmetry_policy=symmetry_symbol,
            distance=distance_fn,
        )

    def query(self, v, n):
        """Query for nearest neighbors with NN-Descent."""
        query_vec = np.asfortranarray(v, dtype=np.float32)
        query_jl = self._to_vector(query_vec)
        result = jl.query(
            self._index,
            self._data,
            query_jl,
            n,
            ef_search=self._ef_search,
        )
        return [int(idx) - 1 for idx in result]

    def query_batch(self, queries, n):
        """Batch query variant that respects ef_search."""
        queries_fortran = np.asfortranarray(queries.T, dtype=np.float32)
        queries_jl = self._to_matrix(queries_fortran)
        results = jl.query(
            self._index,
            self._data,
            queries_jl,
            n,
            ef_search=self._ef_search,
        )
        return [[int(idx) - 1 for idx in result] for result in results]

    def __str__(self):
        return (
            "ManifoldANN-NNDescent("
            f"k={self._k}, max_iterations={self._max_iterations}, "
            f"convergence_threshold={self._convergence_threshold}, "
            f"sample_rate={self._sample_rate}, symmetry={self._symmetry_policy}, "
            f"ef_search={self._ef_search})"
        )

    @staticmethod
    def get_name():
        return "ManifoldANN-NNDescent"
