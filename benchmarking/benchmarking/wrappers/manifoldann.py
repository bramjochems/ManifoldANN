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

    # Per-dim warmup tracking. We compile each index type's hot path against
    # a small dummy matrix of the actual dataset dim before the first timed
    # call so dim-specialised methods don't recompile inside `fit()`.
    # Key: (index_kind, dim). Value: True if warmed.
    _warmed_keys = set()

    # Subclasses set this to opt into per-dim JIT warmup. The string is a
    # stable identifier for the index kind (used as the key in
    # `_warmed_keys`). Subclasses also implement `_warmup(dim)` doing a
    # tiny build + query on `randn(Float32, dim, 64)`.
    _warmup_kind = None

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
            jl.seval("using LinearAlgebra")
            # Create conversion functions once
            ManifoldANNWrapper._to_matrix = jl.seval("x -> Matrix{Float32}(x)")
            ManifoldANNWrapper._to_vector = jl.seval("x -> Vector{Float32}(x)")
            # Helper that converts a batch of Vector{Neighbor} into a
            # Matrix{Int} of shape (k, n_queries). This lets us pull the
            # entire batch over to Python in one numpy view via juliacall,
            # rather than iterating Julia objects through Python `for`
            # loops. Used in `finalize_batch_ids` (called OUTSIDE the
            # timed region — see base.py for the contract).
            jl.seval("""
                function _mann_bench_neighbors_to_id_matrix(batches)
                    n = length(batches)
                    n == 0 && return Matrix{Int}(undef, 0, 0)
                    k = length(batches[1])
                    out = Matrix{Int}(undef, k, n)
                    @inbounds for j in 1:n
                        b = batches[j]
                        for i in 1:k
                            out[i, j] = b[i].id
                        end
                    end
                    return out
                end
            """)
            ManifoldANNWrapper._julia_initialized = True

        self._index = None
        self._data = None

    # ------------------------------------------------------------------
    # Lifecycle: marshal numpy -> Julia outside the timed region
    # ------------------------------------------------------------------

    def prepare_data(self, X):
        """Convert numpy (n_samples, n_features) -> Julia Matrix{Float32}
        of shape (n_features, n_samples). Charged outside `fit()`."""
        X_fortran = np.asfortranarray(X.T, dtype=np.float32)
        prepared = self._to_matrix(X_fortran)

        # Per-dim JIT warmup for this index kind. Done outside the timed
        # region. Only warms once per (kind, dim) across the whole process.
        kind = type(self)._warmup_kind
        if kind is not None:
            key = (kind, int(X.shape[1]))
            if key not in ManifoldANNWrapper._warmed_keys:
                try:
                    self._warmup(int(X.shape[1]))
                except Exception as exc:
                    print(f"  ⚠ {kind} warmup at dim={X.shape[1]} failed: {exc}")
                ManifoldANNWrapper._warmed_keys.add(key)

        return prepared

    def prepare_queries(self, Q):
        """Convert query batch numpy (n_queries, n_features) -> Julia
        Matrix{Float32} of shape (n_features, n_queries)."""
        Q_fortran = np.asfortranarray(Q.T, dtype=np.float32)
        return self._to_matrix(Q_fortran)

    def _warmup(self, dim: int) -> None:
        """Default warmup: subclasses with `_warmup_kind` set should
        override. Builds a tiny instance of the index and runs one batch
        query so dim-specialised code paths compile."""
        pass

    def warmup_build(self, dim: int, n: int = 200) -> None:
        """Build at the *actual* config on a small synthetic dataset.

        Subclasses override and call `fit()` on a small Julia matrix so
        Julia's build-path JIT specialises on the actual integer
        parameters (M, ef_construction, nlist, ...). Costs a few seconds
        once per (algorithm, dim), independent of `--reps`.

        Default fall-through to the cheap dim-only warmup is fine for
        algorithms whose builders don't change shape with their config.
        """
        kind = type(self)._warmup_kind
        if kind is None:
            return
        key = ("warmup_build", kind, int(dim), self._signature_for_warmup())
        if key in ManifoldANNWrapper._warmed_keys:
            return
        try:
            self._warmup_build_actual(dim, n)
        except Exception as exc:
            print(f"  ⚠ {kind} build-warmup at dim={dim} failed: {exc}")
        ManifoldANNWrapper._warmed_keys.add(key)

    def _signature_for_warmup(self):
        """Override to include config knobs that change codegen shape."""
        return ()

    def _warmup_build_actual(self, dim: int, n: int) -> None:
        """Subclasses implement: build the index on `randn(Float32, dim, n)`
        with the wrapper's actual config, then run one batch query.
        """
        return None

    def _neighbors_to_ids(self, jl_neighbors):
        """Convert Julia neighbor structs to 0-indexed Python ids."""
        ids = jl.ManifoldANN.neighbor_ids(jl_neighbors)
        return [int(idx) - 1 for idx in ids]

    def _batch_neighbors_to_ids(self, jl_neighbor_batches):
        """Convert Julia neighbor batches to 0-indexed Python ids.

        Goes through a Julia-side `Matrix{Int}` (k × n_queries) so we
        cross the Julia↔Python boundary once with a contiguous block,
        then numpy-vectorise the 1-based-to-0-based shift.
        """
        id_matrix_jl = jl.seval("Main._mann_bench_neighbors_to_id_matrix")(
            jl_neighbor_batches
        )
        idxs_np = np.asarray(id_matrix_jl, dtype=np.int64)
        # idxs_np has shape (k, n_queries); transpose for per-query rows.
        return (idxs_np - 1).T.tolist()

    def set_num_threads(self, n: int) -> None:
        """Pin BLAS threads on the Julia side so cross-library timings see
        the same threading regime.

        Julia's own thread pool is fixed at process start by
        JULIA_NUM_THREADS; this hook only handles the BLAS-thread knob,
        which several Julia algorithms (k-means inside IVF, linear-algebra
        in spectral preprocessors) read at runtime.
        """
        try:
            jl.seval(f"using LinearAlgebra; BLAS.set_num_threads({int(n)})")
        except Exception:
            pass

    def _get_distance_function(self):
        """Get the appropriate Julia distance function for the metric.

        Returns:
            Julia function for computing distances (matching ground truth computation)
        """
        if self._metric == "angular":
            # Use cosine_distance (1 - cosine_sim) to match ground truth
            return jl.ManifoldANN.cosine_distance
        else:  # euclidean
            # Use Euclidean distance (with sqrt) to match ground truth
            return jl.ManifoldANN.default_distance

    def fit(self, X):
        """Subclasses must override. `X` is the prepared Julia matrix from
        `prepare_data`."""
        # Default: assume `X` is already a Julia matrix and stash it.
        self._data = X

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

        return self._neighbors_to_ids(result)

    def query_batch(self, queries, n):
        """Query for nearest neighbors of multiple queries at once.

        `queries` is the prepared Julia matrix returned from
        `prepare_queries` (or a numpy array, which we convert as a
        fallback). The Python↔Julia boundary is crossed once.
        """
        results_jl = self.query_batch_raw(queries, n)
        return self.finalize_batch_ids(results_jl)

    # Split-timing hooks: keep the timed region focused on the Julia
    # `query` call. `finalize_batch_ids` (id conversion, numpy marshalling)
    # runs OUTSIDE the timed region per the base-class contract.
    def query_batch_raw(self, queries, n):
        if isinstance(queries, np.ndarray):
            queries_fortran = np.asfortranarray(queries.T, dtype=np.float32)
            queries_jl = self._to_matrix(queries_fortran)
        else:
            queries_jl = queries
        return jl.query(self._index, self._data, queries_jl, n)

    def finalize_batch_ids(self, raw):
        return self._batch_neighbors_to_ids(raw)

    @staticmethod
    def is_available():
        """Check if ManifoldANN is available."""
        try:
            from juliacall import Main as jl
            return True
        except ImportError:
            return False


def _warm_data(dim: int, n: int = 64):
    """Build a small Julia warmup matrix of the requested dim."""
    return jl.seval(f"randn(Float32, {dim}, {n})")


def _warm_queries(dim: int, n: int = 4):
    return jl.seval(f"randn(Float32, {dim}, {n})")


class ManifoldANN_BruteForce(ManifoldANNWrapper):
    """Wrapper for ManifoldANN BruteForceIndex (baseline)."""

    _warmup_kind = "BruteForceIndex"

    def __init__(self, metric):
        """Initialize brute force wrapper."""
        super().__init__(metric)

    def _warmup(self, dim: int) -> None:
        data = _warm_data(dim)
        qs = _warm_queries(dim)
        idx = jl.build_index(jl.BruteForceIndex, data, distance=self._get_distance_function())
        jl.query(idx, data, qs, 5)

    def fit(self, X):
        """Build the brute force index. `X` is the prepared Julia matrix."""
        self._data = X
        distance_fn = self._get_distance_function()
        self._index = jl.build_index(jl.BruteForceIndex, self._data, distance=distance_fn)

    def __str__(self):
        return "ManifoldANN-BruteForce()"

    @staticmethod
    def get_name():
        return "ManifoldANN-BruteForce"


class ManifoldANN_LSH(ManifoldANNWrapper):
    """Wrapper for ManifoldANN LSHIndex."""

    _warmup_kind = "LSHIndex"

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

    def _warmup(self, dim: int) -> None:
        data = _warm_data(dim)
        qs = _warm_queries(dim)
        # Warm both hash families against this dim.
        try:
            idx_h = jl.build_index(
                jl.LSHIndex, data,
                n_tables=2, hash_length=4,
                hash_factory=jl.make_random_hyperplane_hash,
                T=jl.Float32,
            )
            jl.query(idx_h, data, qs, 5)
        except Exception:
            pass
        try:
            idx_b = jl.build_index(
                jl.LSHIndex, data,
                n_tables=2, hash_length=4,
                hash_factory=jl.make_binning_hash,
                bin_width=1.0,
                use_offset=True,
                T=jl.Float32,
            )
            jl.query(idx_b, data, qs, 5)
        except Exception:
            pass

    def fit(self, X):
        """Build the LSH index. `X` is the prepared Julia matrix."""
        self._data = X

        # Select appropriate hash family based on metric
        if self._metric == "euclidean":
            # Use BinningHash (p-stable LSH) for Euclidean distance
            if self._bin_width is None:
                # Auto-bin_width is computed from the original numpy data
                # in `prepare_data`. If this hasn't been computed yet (e.g.
                # the harness called `fit` directly with the prepared
                # matrix), fall back to a sensible default.
                self._bin_width = getattr(self, "_bin_width_auto", 1.0)

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

    def prepare_data(self, X):
        """Override to compute auto bin_width from the numpy view (we still
        have access to it here) before handing the matrix to Julia."""
        if self._metric == "euclidean" and self._bin_width is None:
            n_samples = min(1000, X.shape[0])
            indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
            sample = X[indices]

            from scipy.spatial.distance import cdist
            distances = cdist(sample, sample, metric="euclidean")
            np.fill_diagonal(distances, np.inf)
            avg_nn_dist = np.mean(np.min(distances, axis=1))

            self._bin_width_auto = 3.0 * avg_nn_dist
            print(
                f"LSH: Auto-computed bin_width = {self._bin_width_auto:.4f} "
                f"(3 × avg_nn_dist = 3 × {avg_nn_dist:.4f})"
            )
        return super().prepare_data(X)

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

        return self._neighbors_to_ids(result)

    def __str__(self):
        cap_str = (
            f", candidate_cap={self._candidate_cap}"
            if hasattr(self, "_candidate_cap")
            else ""
        )
        bw = self._bin_width if self._bin_width is not None else getattr(self, "_bin_width_auto", None)
        bw_str = f", bin_width={bw:.4f}" if bw is not None else ""
        return f"ManifoldANN-LSH(n_tables={self._n_tables}, hash_length={self._hash_length}{bw_str}{cap_str})"

    @staticmethod
    def get_name():
        return "ManifoldANN-LSH"


class ManifoldANN_KDTree(ManifoldANNWrapper):
    """Wrapper for ManifoldANN KDTreeIndex."""

    _VALID_AXIS_SELECTORS = ("variance", "cyclic")
    _warmup_kind = "KDTreeIndex"

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

    def _warmup(self, dim: int) -> None:
        data = _warm_data(dim)
        warm_q = jl.seval(f"randn(Float32, {dim})")
        for sel in ("variance", "cyclic"):
            try:
                idx = jl.build_index(jl.KDTreeIndex, data, axis_selector=jl.Symbol(sel))
                jl.query(idx, data, warm_q, 5)
            except Exception:
                pass

    def fit(self, X):
        """Build the KD-tree index. `X` is the prepared Julia matrix."""
        self._data = X

        axis_symbol = jl.Symbol(self._axis_selector)
        self._index = jl.build_index(
            jl.KDTreeIndex,
            self._data,
            axis_selector=axis_symbol,
        )

    # KDTreeIndex now reaches batch via the generic AbstractANNIndex
    # matrix-dispatch path (threaded internally), so we use the inherited
    # `query_batch_raw` — no Python-side loop.

    def __str__(self):
        return f"ManifoldANN-KDTree(axis_selector={self._axis_selector})"

    @staticmethod
    def get_name():
        return "ManifoldANN-KDTree"


class ManifoldANN_HNSW(ManifoldANNWrapper):
    """Wrapper for ManifoldANN HNSWIndex."""

    _VALID_NEIGHBOR_POLICIES = {"heuristic", "diversified"}
    _warmup_kind = "HNSWIndex"

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

    def _warmup(self, dim: int) -> None:
        data = _warm_data(dim, n=128)
        qs = _warm_queries(dim, n=4)
        warm_q = jl.seval(f"randn(Float32, {dim})")
        distance_fn = self._get_distance_function()
        for policy in ("heuristic", "diversified"):
            try:
                idx = jl.build_index(
                    jl.HNSWIndex, data,
                    M=8, ef_construction=40, ef_search=16,
                    neighbor_policy=jl.Symbol(policy),
                    distance=distance_fn,
                )
                jl.query(idx, data, warm_q, 5, ef_search=16)
                jl.query(idx, data, qs, 5, ef_search=16)
            except Exception:
                pass

    def _signature_for_warmup(self):
        return (self._M, self._ef_construction, self._neighbor_policy)

    def _warmup_build_actual(self, dim: int, n: int) -> None:
        data = jl.seval(f"randn(Float32, {dim}, {n})")
        qs = jl.seval(f"randn(Float32, {dim}, 16)")
        idx = jl.build_index(
            jl.HNSWIndex, data,
            M=self._M, ef_construction=self._ef_construction,
            ef_search=self._ef_search,
            neighbor_policy=jl.Symbol(self._neighbor_policy),
            distance=self._get_distance_function(),
        )
        jl.query(idx, data, qs, 5, ef_search=self._ef_search)

    def fit(self, X):
        """Build the HNSW index. `X` is the prepared Julia matrix."""
        self._data = X

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

        return self._neighbors_to_ids(result)

    def query_batch_raw(self, queries, n):
        """Batch HNSW query, respecting `ef_search`."""
        if isinstance(queries, np.ndarray):
            queries_fortran = np.asfortranarray(queries.T, dtype=np.float32)
            queries_jl = self._to_matrix(queries_fortran)
        else:
            queries_jl = queries
        return jl.query(
            self._index, self._data, queries_jl, n, ef_search=self._ef_search
        )

    def __str__(self):
        return (
            f"ManifoldANN-HNSW(M={self._M}, ef_construction={self._ef_construction}, "
            f"ef_search={self._ef_search}, neighbor_policy={self._neighbor_policy})"
        )

    @staticmethod
    def get_name():
        return "ManifoldANN-HNSW"


class ManifoldANN_IVFHNSW(ManifoldANNWrapper):
    """Wrapper for the IVF (KMeans) + HNSW multi-level index."""

    _warmup_kind = "IVFHNSW"

    def __init__(
        self,
        metric,
        nlist=64,
        routing_k=8,
        kmeans_init="kmeans_plus_plus",
        kmeans_max_iters=50,
        kmeans_tol=1e-4,
        M=16,
        ef_construction=200,
        ef_search=64,
        neighbor_policy="diversified",
    ):
        super().__init__(metric)
        self._nlist = int(nlist)
        self._routing_k = int(routing_k)
        self._kmeans_init = kmeans_init
        self._kmeans_max_iters = int(kmeans_max_iters)
        self._kmeans_tol = float(kmeans_tol)
        self._M = int(M)
        self._ef_construction = int(ef_construction)
        self._ef_search = int(ef_search)
        self._neighbor_policy = neighbor_policy

    def _kmeans_metric(self):
        if self._metric == "angular":
            return jl.seval("ManifoldANN.Distances.CosineDist()")
        return jl.seval("ManifoldANN.Distances.Euclidean()")

    def _warmup(self, dim: int) -> None:
        data = _warm_data(dim, n=128)
        warm_q = jl.seval(f"randn(Float32, {dim})")
        try:
            idx = jl.build_ivf_hnsw_index(
                data,
                nlist=4,
                routing_k=2,
                kmeans_distance=self._kmeans_metric(),
                kmeans_init=jl.Symbol(self._kmeans_init),
                kmeans_max_iters=2,
                kmeans_tol=self._kmeans_tol,
                hnsw_M=8,
                hnsw_ef_construction=40,
                hnsw_ef_search=16,
                hnsw_neighbor_policy=jl.Symbol(self._neighbor_policy),
                distance=self._get_distance_function(),
            )
            jl.query(idx, data, warm_q, 5)
        except Exception:
            pass

    def fit(self, X):
        """Build the IVF+HNSW index. `X` is the prepared Julia matrix."""
        self._data = X
        distance_fn = self._get_distance_function()
        routing_k = max(1, min(self._routing_k, self._nlist))
        kmeans_metric = self._kmeans_metric()
        self._index = jl.build_ivf_hnsw_index(
            self._data,
            nlist=self._nlist,
            routing_k=routing_k,
            kmeans_distance=kmeans_metric,
            kmeans_init=jl.Symbol(self._kmeans_init),
            kmeans_max_iters=self._kmeans_max_iters,
            kmeans_tol=self._kmeans_tol,
            hnsw_M=self._M,
            hnsw_ef_construction=self._ef_construction,
            hnsw_ef_search=self._ef_search,
            hnsw_neighbor_policy=jl.Symbol(self._neighbor_policy),
            distance=distance_fn,
        )

    def __str__(self):
        return (
            "ManifoldANN-IVF-HNSW("
            f"nlist={self._nlist}, routing_k={self._routing_k}, "
            f"M={self._M}, ef_construction={self._ef_construction}, "
            f"ef_search={self._ef_search}, neighbor_policy={self._neighbor_policy}, "
            f"kmeans_init={self._kmeans_init})"
        )

    @staticmethod
    def get_name():
        return "ManifoldANN-IVF-HNSW"


class ManifoldANN_IVFFlat(ManifoldANNWrapper):
    """Wrapper for the Julia IVF-Flat index."""

    _warmup_kind = "IVFFlatIndex"

    def __init__(
        self,
        metric,
        nlist=100,
        nprobe=10,
        kmeans_init="kmeans_plus_plus",
        kmeans_max_iters=5,
        kmeans_tol=1e-4,
    ):
        super().__init__(metric)
        self._nlist = int(nlist)
        self._nprobe = int(nprobe)
        self._kmeans_init = kmeans_init
        self._kmeans_max_iters = int(kmeans_max_iters)
        self._kmeans_tol = float(kmeans_tol)

    def _kmeans_metric(self):
        if self._metric == "angular":
            return jl.seval("ManifoldANN.Distances.CosineDist()")
        return jl.seval("ManifoldANN.Distances.Euclidean()")

    def _warmup(self, dim: int) -> None:
        data = _warm_data(dim, n=128)
        warm_q = jl.seval(f"randn(Float32, {dim})")
        qs = _warm_queries(dim)
        try:
            idx = jl.build_index(
                jl.IVFFlatIndex, data,
                nlist=4, nprobe=2,
                distance=self._get_distance_function(),
                centroid_metric=self._kmeans_metric(),
            )
            jl.query(idx, data, warm_q, 5, nprobe=2)
            jl.query(idx, data, qs, 5, nprobe=2)
        except Exception:
            pass

    def fit(self, X):
        """Build the IVF-Flat index. `X` is the prepared Julia matrix."""
        self._data = X
        distance_fn = self._get_distance_function()
        kmeans_metric = self._kmeans_metric()
        self._index = jl.build_index(
            jl.IVFFlatIndex,
            self._data,
            nlist=self._nlist,
            nprobe=self._nprobe,
            distance=distance_fn,
            centroid_metric=kmeans_metric,
        )

    def query(self, v, n):
        """Query IVF-Flat."""
        query_vec = np.asfortranarray(v, dtype=np.float32)
        query_jl = self._to_vector(query_vec)
        result = jl.query(self._index, self._data, query_jl, n, nprobe=self._nprobe)
        return self._neighbors_to_ids(result)

    def query_batch_raw(self, queries, n):
        if isinstance(queries, np.ndarray):
            queries_fortran = np.asfortranarray(queries.T, dtype=np.float32)
            queries_jl = self._to_matrix(queries_fortran)
        else:
            queries_jl = queries
        return jl.query(self._index, self._data, queries_jl, n, nprobe=self._nprobe)

    def __str__(self):
        return (
            "ManifoldANN-IVF-Flat("
            f"nlist={self._nlist}, nprobe={self._nprobe}, "
            f"kmeans_init={self._kmeans_init}, kmeans_max_iters={self._kmeans_max_iters})"
        )

    @staticmethod
    def get_name():
        return "ManifoldANN-IVF-Flat"


class ManifoldANN_NNDescent(ManifoldANNWrapper):
    """Wrapper for ManifoldANN NNDescentIndex."""

    _warmup_kind = "NNDescentIndex"

    def __init__(
        self,
        metric,
        k=32,
        max_iterations=5,
        convergence_threshold=0.01,
        sample_rate=0.5,
        symmetry_policy="pruned",
        apply_symmetry_continuously=False,
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
            apply_symmetry_continuously: Apply symmetry after each iteration (True) or only at end (False, default)
            ef_search: Beam width for graph search queries (default: 2*k)
        """
        super().__init__(metric)
        self._k = int(k)
        self._max_iterations = int(max_iterations)
        self._convergence_threshold = float(convergence_threshold)
        self._sample_rate = float(sample_rate)
        self._symmetry_policy = symmetry_policy
        self._apply_symmetry_continuously = bool(apply_symmetry_continuously)
        self._ef_search = ef_search if ef_search is not None else max(self._k, 2 * self._k)

    def _warmup(self, dim: int) -> None:
        data = _warm_data(dim, n=128)
        warm_q = jl.seval(f"randn(Float32, {dim})")
        qs = _warm_queries(dim)
        sampling = jl.ManifoldANN.UniformPairSampling(0.5)
        for symmetry in ("pruned", "full"):
            for cont in (True, False):
                try:
                    idx = jl.build_index(
                        jl.NNDescentIndex, data,
                        k=8, max_iterations=2, convergence_threshold=0.1,
                        sampling_policy=sampling,
                        symmetry_policy=jl.Symbol(symmetry),
                        apply_symmetry_continuously=cont,
                        distance=self._get_distance_function(),
                    )
                    jl.query(idx, data, warm_q, 5, ef_search=16)
                    jl.query(idx, data, qs, 5, ef_search=16)
                except Exception:
                    pass

    def _signature_for_warmup(self):
        return (
            self._k, self._max_iterations, self._symmetry_policy,
            self._apply_symmetry_continuously,
        )

    def _warmup_build_actual(self, dim: int, n: int) -> None:
        data = jl.seval(f"randn(Float32, {dim}, {n})")
        qs = jl.seval(f"randn(Float32, {dim}, 16)")
        sampling = jl.ManifoldANN.UniformPairSampling(self._sample_rate)
        idx = jl.build_index(
            jl.NNDescentIndex, data,
            k=self._k, max_iterations=self._max_iterations,
            convergence_threshold=self._convergence_threshold,
            sampling_policy=sampling,
            symmetry_policy=jl.Symbol(self._symmetry_policy),
            apply_symmetry_continuously=self._apply_symmetry_continuously,
            distance=self._get_distance_function(),
        )
        jl.query(idx, data, qs, 5, ef_search=self._ef_search)

    def fit(self, X):
        """Build the NN-Descent index. `X` is the prepared Julia matrix."""
        self._data = X
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
            apply_symmetry_continuously=self._apply_symmetry_continuously,
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
        return self._neighbors_to_ids(result)

    def query_batch_raw(self, queries, n):
        """Batch query variant that respects ef_search."""
        if isinstance(queries, np.ndarray):
            queries_fortran = np.asfortranarray(queries.T, dtype=np.float32)
            queries_jl = self._to_matrix(queries_fortran)
        else:
            queries_jl = queries
        return jl.query(
            self._index,
            self._data,
            queries_jl,
            n,
            ef_search=self._ef_search,
        )

    def __str__(self):
        symmetry_mode = "continuous" if self._apply_symmetry_continuously else "deferred"
        return (
            "ManifoldANN-NNDescent("
            f"k={self._k}, max_iterations={self._max_iterations}, "
            f"convergence_threshold={self._convergence_threshold}, "
            f"sample_rate={self._sample_rate}, symmetry={self._symmetry_policy} ({symmetry_mode}), "
            f"ef_search={self._ef_search})"
        )

    @staticmethod
    def get_name():
        return "ManifoldANN-NNDescent"
