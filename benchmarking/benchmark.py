#!/usr/bin/env python3
"""
Combined comparison test: ManifoldANN vs ann-benchmarks algorithms.

This script compares your Julia implementation against other popular
ANN algorithms from the ann-benchmarks suite on the same datasets.

This gives you a direct "apples-to-apples" comparison.
"""

import time
import numpy as np
import h5py
import os
import sys
from urllib.request import urlretrieve

# Configure Julia threading BEFORE importing juliacall
# Default to using all available threads if not specified
if "JULIA_NUM_THREADS" not in os.environ:
    import multiprocessing

    os.environ["JULIA_NUM_THREADS"] = str(multiprocessing.cpu_count())
    print(
        f"ℹ️  Setting JULIA_NUM_THREADS={os.environ['JULIA_NUM_THREADS']} (auto-detected)"
    )

# Suppress juliacall threading warning (we know what we're doing)
if "PYTHON_JULIACALL_HANDLE_SIGNALS" not in os.environ:
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

BASE_DIR = os.path.dirname(__file__)
ANN_BENCHMARKS_DIR = os.path.join(BASE_DIR, "ann-benchmarks")

sys.path.insert(0, BASE_DIR)
from manifoldann_wrapper import (
    LSHWrapper,
    HNSWWrapper as ManifoldHNSW,
    BruteForceWrapper as ManifoldBruteForce,
    KDTreeWrapper as ManifoldKDTree,
    NNDescentWrapper as ManifoldNNDescent,
)

# Import ann-benchmarks algorithms if available
try:
    from annoy import AnnoyIndex

    ANNOY_AVAILABLE = True
except ImportError:
    ANNOY_AVAILABLE = False
    print("⚠️  Annoy not available. Install with: pip install annoy")

try:
    import hnswlib

    HNSWLIB_AVAILABLE = True
except ImportError:
    HNSWLIB_AVAILABLE = False
    print("⚠️  HNSWlib not available. Install with: pip install hnswlib")

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not available. Install with: pip install faiss-cpu")

try:
    from scipy.spatial import cKDTree

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  SciPy not available. Install with: pip install scipy")


class AnnoyWrapper:
    """Wrapper for Annoy to match our interface."""

    def __init__(self, metric, n_trees=10):
        self.metric = "angular" if metric == "angular" else "euclidean"
        self.n_trees = n_trees
        self.index = None
        self.dimension = None

    def fit(self, X):
        self.dimension = X.shape[1]
        self.index = AnnoyIndex(self.dimension, self.metric)
        for i, vec in enumerate(X):
            self.index.add_item(i, vec.tolist())
        self.index.build(self.n_trees)

    def query(self, v, n):
        return self.index.get_nns_by_vector(v.tolist(), n)

    def __str__(self):
        return f"Annoy(n_trees={self.n_trees})"


class HNSWWrapper:
    """Wrapper for HNSW to match our interface."""

    def __init__(self, metric, M=16, ef_construction=200):
        self.metric = "cosine" if metric == "angular" else "l2"
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = 50  # Will be set by set_query_arguments
        self.index = None
        self.dimension = None

    def fit(self, X):
        self.dimension = X.shape[1]
        self.index = hnswlib.Index(space=self.metric, dim=self.dimension)
        self.index.init_index(
            max_elements=len(X), ef_construction=self.ef_construction, M=self.M
        )

        # Set thread count to match JULIA_NUM_THREADS if specified
        if "JULIA_NUM_THREADS" in os.environ:
            num_threads = int(os.environ["JULIA_NUM_THREADS"])
            self.index.set_num_threads(num_threads)

        self.index.add_items(X)

    def set_query_arguments(self, ef_search=50):
        self.ef_search = ef_search
        if self.index:
            self.index.set_ef(ef_search)

    def query(self, v, n):
        if self.index:
            self.index.set_ef(self.ef_search)
        labels, distances = self.index.knn_query(v.reshape(1, -1), k=n)
        return labels[0].tolist()

    def __str__(self):
        return f"HNSW(M={self.M}, ef={self.ef_search})"


class FAISSWrapper:
    """Wrapper for FAISS HNSW to match our interface."""

    def __init__(self, metric, M=16, ef_construction=200):
        self.metric = metric
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = 50  # Will be set by set_query_arguments
        self.index = None
        self.dimension = None

    def fit(self, X):
        self.dimension = X.shape[1]

        # Create HNSW index
        if self.metric == "angular":
            # For angular/cosine distance, normalize vectors and use L2
            # (cosine similarity = 1 - ||normalized(a) - normalized(b)||^2 / 2)
            X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
            self.index = faiss.IndexHNSWFlat(self.dimension, self.M, faiss.METRIC_L2)
            self.X_normalized = X_normalized
        else:
            # Euclidean distance
            self.index = faiss.IndexHNSWFlat(self.dimension, self.M, faiss.METRIC_L2)
            self.X_normalized = None

        # Set construction parameters
        self.index.hnsw.efConstruction = self.ef_construction

        # Set thread count to match JULIA_NUM_THREADS if specified
        if "JULIA_NUM_THREADS" in os.environ:
            num_threads = int(os.environ["JULIA_NUM_THREADS"])
            faiss.omp_set_num_threads(num_threads)

        # Add data
        if self.X_normalized is not None:
            self.index.add(self.X_normalized.astype("float32"))
        else:
            self.index.add(X.astype("float32"))

    def set_query_arguments(self, ef_search=50):
        self.ef_search = ef_search
        if self.index:
            self.index.hnsw.efSearch = ef_search

    def query(self, v, n):
        if self.index:
            self.index.hnsw.efSearch = self.ef_search

        if self.X_normalized is not None:
            v_normalized = v / np.linalg.norm(v)
            v_query = v_normalized.reshape(1, -1).astype("float32")
        else:
            v_query = v.reshape(1, -1).astype("float32")

        distances, labels = self.index.search(v_query, n)
        return labels[0].tolist()

    def __str__(self):
        return f"FAISS-HNSW(M={self.M}, ef={self.ef_search})"


class KDTreeWrapper:
    """Wrapper for SciPy's cKDTree."""

    def __init__(self, metric, leafsize=40):
        self.metric = metric
        self.leafsize = leafsize
        self.tree = None
        self.data = None

    def _maybe_normalize(self, X):
        if self.metric != "angular":
            return X
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms

    def _normalize_vector(self, v):
        if self.metric != "angular":
            return v
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def fit(self, X):
        data = self._maybe_normalize(X.astype(np.float64, copy=False))
        self.tree = cKDTree(data, leafsize=self.leafsize)
        self.data = data

    def query(self, v, n):
        if self.tree is None:
            raise RuntimeError("KDTree has not been built yet. Call fit() first.")
        query_vec = self._normalize_vector(v.astype(np.float64, copy=False))
        distances, indices = self.tree.query(query_vec, k=n)

        if np.isscalar(indices):
            return [int(indices)]
        return [int(idx) for idx in np.atleast_1d(indices)]

    def __str__(self):
        return f"KDTree(leafsize={self.leafsize}, metric={self.metric})"


def download_dataset(dataset_name, data_dir="data"):
    """Download dataset if not already present."""
    os.makedirs(data_dir, exist_ok=True)
    dataset_path = os.path.join(data_dir, f"{dataset_name}.hdf5")

    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return dataset_path

    url = f"https://ann-benchmarks.com/{dataset_name}.hdf5"
    print(f"Downloading {dataset_name}...")
    print("This may take a few minutes...")

    try:
        urlretrieve(url, dataset_path)
        print("Download complete!")
        return dataset_path
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def load_dataset(dataset_path, max_train=None):
    """Load dataset from HDF5 file."""
    print(f"\nLoading dataset from {dataset_path}...")
    with h5py.File(dataset_path, "r") as f:
        train = np.array(f["train"])
        test = np.array(f["test"])
        neighbors = np.array(f["neighbors"])
        distance_metric = f.attrs.get("distance", "unknown")

        print(f"Original train shape: {train.shape}")
        print(f"Test shape: {test.shape}")
        print(f"Distance metric: {distance_metric}")

        if max_train and train.shape[0] > max_train:
            print(
                f"\n⚠️  Limiting train set from {train.shape[0]} to {max_train} samples"
            )
            print(f"   Recomputing ground truth neighbors for limited dataset...")
            train = train[:max_train]

            # Recompute ground truth for the limited dataset
            # This is necessary because original ground truth references full dataset
            # Use vectorized operations for ~100x speedup
            print(f"   Computing ground truth using vectorized operations...")
            k = neighbors.shape[1]  # Use same k as original

            if distance_metric == "euclidean":
                # Vectorized: (n_test, n_train) distance matrix
                # Use broadcasting: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
                test_norm_sq = np.sum(test**2, axis=1, keepdims=True)  # (n_test, 1)
                train_norm_sq = np.sum(train**2, axis=1)  # (n_train,)
                distances = (
                    test_norm_sq + train_norm_sq - 2 * test @ train.T
                )  # (n_test, n_train)
                distances = np.sqrt(np.maximum(distances, 0))  # Avoid numerical issues
            elif distance_metric == "angular":
                # Vectorized: cosine distance = 1 - cosine_similarity
                # Compute cosine similarity with proper normalization
                # cosine_sim = (x · y) / (||x|| * ||y||)

                # Normalize test and train vectors
                test_norms = np.linalg.norm(test, axis=1, keepdims=True)
                train_norms = np.linalg.norm(train, axis=1, keepdims=True)

                # Avoid division by zero
                test_norms[test_norms == 0] = 1.0
                train_norms[train_norms == 0] = 1.0

                test_normalized = test / test_norms
                train_normalized = train / train_norms

                # Compute cosine similarity
                similarities = test_normalized @ train_normalized.T  # (n_test, n_train)
                distances = 1 - similarities
            else:
                raise ValueError(f"Unknown distance metric: {distance_metric}")

            # Get k nearest neighbors for all queries at once
            # Use argpartition for O(n) complexity instead of O(n log n) sorting
            neighbors_recomputed = np.argpartition(
                distances, min(k, distances.shape[1] - 1), axis=1
            )[:, :k]

            # Sort the k nearest neighbors by distance
            for i in range(neighbors_recomputed.shape[0]):
                neighbors_recomputed[i] = neighbors_recomputed[
                    i, np.argsort(distances[i, neighbors_recomputed[i]])
                ]

            neighbors = neighbors_recomputed
            print(f"   Ground truth recomputed for {max_train} train samples")

        print(f"Final train shape: {train.shape}")
        print(f"Ground truth shape: {neighbors.shape}")

        # Sanity check: verify ground truth by checking first query's nearest neighbor
        if len(neighbors) > 0 and len(train) > 0:
            first_query = test[0]
            expected_nn = neighbors[0][0]  # First nearest neighbor

            if distance_metric == "euclidean":
                dist_to_nn = np.linalg.norm(first_query - train[expected_nn])
                print(f"Sanity check: distance to 1st NN = {dist_to_nn:.4f}")
            elif distance_metric == "angular":
                # Compute cosine distance
                q_norm = np.linalg.norm(first_query)
                nn_norm = np.linalg.norm(train[expected_nn])
                if q_norm > 0 and nn_norm > 0:
                    cos_sim = np.dot(first_query, train[expected_nn]) / (q_norm * nn_norm)
                    cos_dist = 1 - cos_sim
                    print(f"Sanity check: cosine distance to 1st NN = {cos_dist:.4f}")

        return train, test, neighbors


def ensure_ann_benchmarks_checkout():
    """Warn if the local ann-benchmarks checkout is missing."""
    if not os.path.isdir(ANN_BENCHMARKS_DIR):
        script_name = os.path.join(BASE_DIR, "fetch_ann_benchmarks.sh")
        print(
            f"\n⚠️  ann-benchmarks checkout not found at {ANN_BENCHMARKS_DIR}."
            f"\n   Run {script_name} to clone the pinned repository (ignored by git)."
        )


def compute_recall_at_k(predicted, ground_truth, k):
    """Compute recall@k metric."""
    predicted_set = set(predicted[:k])
    ground_truth_set = set(ground_truth[:k])
    return len(predicted_set & ground_truth_set) / k


def test_algorithm(wrapper, X_train, X_test, ground_truth, k=10, n_queries=100):
    """Test an algorithm and report metrics."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {wrapper}")
    print(f"{'=' * 60}")

    X_test = X_test[:n_queries]
    ground_truth = ground_truth[:n_queries]

    # Build index
    print("Building index...")
    start = time.perf_counter()
    wrapper.fit(X_train)
    build_time = time.perf_counter() - start
    print(f"Build time: {build_time:.2f} seconds")

    # Set query arguments if available
    if hasattr(wrapper, "set_query_arguments"):
        wrapper.set_query_arguments()

    # Query
    print(f"Running {n_queries} queries...")
    query_times = []
    recalls = []

    # Use batch querying if available (much faster for Julia wrappers)
    if hasattr(wrapper, "query_batch"):
        print(f"  Using batch query interface for better performance...")

        # Warmup: Run a few queries to trigger JIT compilation
        if hasattr(wrapper, "_julia_initialized"):
            print(f"  Running warmup queries (Julia JIT compilation)...")
            warmup_start = time.perf_counter()
            _ = wrapper.query_batch(X_test[:5], k)
            warmup_time = time.perf_counter() - warmup_start
            print(f"  Warmup took {warmup_time:.2f}s")

        start = time.perf_counter()
        try:
            results_batch = wrapper.query_batch(X_test, k)
            total_time = time.perf_counter() - start

            # Compute per-query time and recall
            for i, (predicted, true_neighbors) in enumerate(
                zip(results_batch, ground_truth)
            ):
                query_times.append(total_time / n_queries)  # Average time
                recall = compute_recall_at_k(predicted, true_neighbors, k)
                recalls.append(recall)

            print(
                f"  Batch query completed: {total_time:.4f}s total, {total_time / n_queries * 1000:.4f}ms per query"
            )
        except Exception as e:
            print(f"  Error in batch query: {e}")
            import traceback

            traceback.print_exc()
    else:
        # Fall back to individual queries
        for i, (query, true_neighbors) in enumerate(zip(X_test, ground_truth)):
            start = time.perf_counter()
            try:
                predicted = wrapper.query(query, k)
                query_time = time.perf_counter() - start

                query_times.append(query_time)
                recall = compute_recall_at_k(predicted, true_neighbors, k)
                recalls.append(recall)

                if (i + 1) % 20 == 0:
                    print(f"  Processed {i + 1}/{n_queries} queries...")
            except Exception as e:
                print(f"  Error on query {i}: {e}")
                continue

    if not query_times:
        print("  No successful queries!")
        return None

    # Compute statistics
    avg_query_time_ms = np.mean(query_times) * 1000
    std_query_time_ms = np.std(query_times) * 1000
    avg_recall = np.mean(recalls)
    qps = len(query_times) / sum(query_times)

    # Report results
    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"{'=' * 60}")
    print(f"Build time:          {build_time:.2f} s")
    print(f"Avg query time:      {avg_query_time_ms:.4f} ± {std_query_time_ms:.4f} ms")
    print(f"Queries per second:  {qps:.1f}")
    print(f"Recall@{k}:          {avg_recall:.4f} ({avg_recall * 100:.2f}%)")
    print(f"{'=' * 60}")

    return {
        "name": str(wrapper),
        "build_time": build_time,
        "avg_query_time_ms": avg_query_time_ms,
        "qps": qps,
        "recall": avg_recall,
        "implementation": "ManifoldANN"
        if isinstance(wrapper, (LSHWrapper, ManifoldHNSW, ManifoldBruteForce, ManifoldKDTree, ManifoldNNDescent))
        else "Other",
    }


# Dataset configuration: maps short names to (full_name, metric)
DATASET_CONFIG = {
    "fashion-mnist": ("fashion-mnist-784-euclidean", "euclidean"),
    "glove-25": ("glove-25-angular", "angular"),
    "glove-50": ("glove-50-angular", "angular"),
    "glove-100": ("glove-100-angular", "angular"),
    "lastfm": ("lastfm-64-angular", "angular"),
    "mnist": ("mnist-784-euclidean", "euclidean"),
    "nytimes": ("nytimes-256-angular", "angular"),
    "sift": ("sift-128-euclidean", "euclidean"),
}


def main():
    """Run comparison benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare ManifoldANN with ann-benchmarks algorithms"
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASET_CONFIG.keys()),
        default="fashion-mnist",
        help="Which dataset to use",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=10000,
        help="Limit training set size (default: 10K, use 0 for full dataset)",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=100,
        help="Number of test queries (default: 100)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of threads (default: all cores). Applies to both ManifoldANN and hnswlib.",
    )
    args = parser.parse_args()

    # Override thread count if specified
    if args.threads is not None:
        import multiprocessing

        os.environ["JULIA_NUM_THREADS"] = str(args.threads)
        print(f"ℹ️  Using {args.threads} threads for all algorithms")

    # Get dataset configuration
    dataset_name, metric = DATASET_CONFIG[args.dataset]

    print("=" * 60)
    print(f"ManifoldANN vs ann-benchmarks Comparison")
    print(f"Dataset: {dataset_name}")
    print(f"Metric: {metric}")
    print("=" * 60)

    ensure_ann_benchmarks_checkout()

    # Download and load dataset
    dataset_path = download_dataset(dataset_name)
    max_train = None if args.max_train == 0 else args.max_train
    X_train, X_test, ground_truth = load_dataset(dataset_path, max_train=max_train)

    k = 10  # Number of neighbors

    # Build algorithm list
    algorithms = [
        # ManifoldANN algorithms
        ManifoldBruteForce(metric=metric),
        ManifoldKDTree(metric=metric, axis_selector="variance"),
        LSHWrapper(metric=metric, n_tables=8, hash_length=16),
        ManifoldHNSW(
            metric=metric,
            M=16,
            ef_construction=200,
            ef_search=50,
            neighbor_policy="heuristic",
        ),
        ManifoldHNSW(
            metric=metric,
            M=16,
            ef_construction=200,
            ef_search=50,
            neighbor_policy="diversified",
        ),
        ManifoldNNDescent(
            metric=metric,
            k=32,
            max_iterations=10,
            convergence_threshold=0.01,  # Stop at 1% improvement (faster, still good recall)
            sample_rate=0.5,  # Sample 50% of pairs (2x faster build)
            symmetry_policy="pruned",  # PyNNDescent-style pruned symmetry (1.5x degree)
            ef_search=64,
        ),
    ]

    # Add external algorithms if available
    if ANNOY_AVAILABLE:
        algorithms.append(AnnoyWrapper(metric=metric, n_trees=10))
        algorithms.append(AnnoyWrapper(metric=metric, n_trees=20))

    if HNSWLIB_AVAILABLE:
        hnsw = HNSWWrapper(metric=metric, M=16, ef_construction=200)
        hnsw.set_query_arguments(ef_search=50)
        algorithms.append(hnsw)

    if FAISS_AVAILABLE:
        faiss_hnsw = FAISSWrapper(metric=metric, M=16, ef_construction=200)
        faiss_hnsw.set_query_arguments(ef_search=50)
        algorithms.append(faiss_hnsw)

    if SCIPY_AVAILABLE:
        algorithms.append(KDTreeWrapper(metric=metric, leafsize=40))

    if not (ANNOY_AVAILABLE or HNSWLIB_AVAILABLE or FAISS_AVAILABLE or SCIPY_AVAILABLE):
        print("\n⚠️  No external algorithms available for comparison!")
        print("Install with: pip install annoy hnswlib faiss-cpu scipy")
        print("Continuing with ManifoldANN-only tests...\n")

    # Run tests
    results = []
    for algo in algorithms:
        try:
            result = test_algorithm(
                algo, X_train, X_test, ground_truth, k=k, n_queries=args.n_queries
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Error testing {algo}: {e}")
            import traceback

            traceback.print_exc()

    # Summary comparison
    if len(results) > 0:
        print(f"\n{'=' * 140}")
        print("FINAL COMPARISON")
        print(f"{'=' * 140}")
        print(
            f"{'Algorithm':<100} {'Impl':<15} {'Build(s)':>10} {'QPS':>10} {'Recall@10':>10}"
        )
        print(f"{'-' * 120}")

        # Sort by recall descending
        results_sorted = sorted(results, key=lambda x: x["recall"], reverse=True)

        for r in results_sorted:
            impl_marker = "🔵" if r["implementation"] == "ManifoldANN" else "⚪"
            print(
                f"{impl_marker} {r['name']:<98} "
                f"{r['implementation']:<15} "
                f"{r['build_time']:>10.2f} "
                f"{r['qps']:>10.1f} "
                f"{r['recall']:>10.4f}"
            )

        print("\n✓ Comparison complete!")

    else:
        print("\n❌ No successful test results")


if __name__ == "__main__":
    main()
