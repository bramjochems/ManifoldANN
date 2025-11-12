#!/usr/bin/env python3
"""
ManifoldANN Benchmarking Suite.

Compare ManifoldANN algorithms against other popular ANN libraries
using standardized datasets and metrics.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np

# Configure Julia threading BEFORE importing juliacall
if "JULIA_NUM_THREADS" not in os.environ:
    import multiprocessing
    os.environ["JULIA_NUM_THREADS"] = str(multiprocessing.cpu_count())
    print(f"ℹ️  Setting JULIA_NUM_THREADS={os.environ['JULIA_NUM_THREADS']} (auto-detected)")

# Suppress juliacall threading warning
if "PYTHON_JULIACALL_HANDLE_SIGNALS" not in os.environ:
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

# Add benchmarking package to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarking.utils import (
    load_config,
    load_algorithm_metadata,
    download_dataset,
    load_dataset,
    compute_recall_batch,
)
from benchmarking.registry import create_algorithm, list_available_algorithms


def get_algorithm_metadata(algo_name: str, metadata_dict: dict) -> dict:
    """Get metadata for an algorithm, using prefix matching for variants.

    Args:
        algo_name: Full algorithm name (e.g., "ManifoldANN-HNSW-heuristic")
        metadata_dict: Dictionary of algorithm metadata from algorithms.yaml

    Returns:
        Metadata dictionary for the algorithm, or empty dict if not found
    """
    # Try exact match first
    if algo_name in metadata_dict:
        return metadata_dict[algo_name]

    # Try prefix match for variants (e.g., "ManifoldANN-HNSW-heuristic" -> "ManifoldANN-HNSW")
    for base_name, metadata in metadata_dict.items():
        if algo_name.startswith(base_name + "-"):
            return metadata

    return {}


def run_benchmark(config_name: str, data_dir: str = "data", k: int = 10):
    """Run benchmarks for a single dataset configuration.

    Args:
        config_name: Name of the config file (without .yaml extension)
        data_dir: Directory for downloading/storing datasets
        k: Number of neighbors to retrieve
    """
    # Load configuration and algorithm metadata
    print("=" * 80)
    print(f"Loading configuration: {config_name}")
    print("=" * 80)
    config = load_config(config_name)
    algo_metadata = load_algorithm_metadata()

    dataset_name = config["dataset"]
    metric = config["metric"]
    n_train = config["n_train"]
    n_test = config["n_test"]

    print(f"Dataset: {dataset_name}")
    print(f"Metric: {metric}")
    print(f"Training points: {n_train}")
    print(f"Test queries: {n_test}")
    print(f"k (neighbors): {k}")

    # Download and load dataset
    dataset_path = download_dataset(dataset_name, data_dir)
    train, test, ground_truth = load_dataset(dataset_path, n_train=n_train, n_test=n_test)

    print(f"\n{'=' * 80}")
    print("Building and Evaluating Algorithms")
    print(f"{'=' * 80}\n")

    results = []

    # Iterate through configured algorithms
    for algo_name, algo_params in config["algorithms"].items():
        print(f"\n{'─' * 80}")
        print(f"Algorithm: {algo_name}")
        print(f"{'─' * 80}")

        # Create algorithm instance
        algo = create_algorithm(algo_name, metric, algo_params or {})

        if algo is None:
            print(f"⚠️  Skipped (library not available)")
            continue

        print(f"Configuration: {algo}")

        try:
            # Build index
            print("Building index...")
            build_start = time.time()
            algo.fit(train)
            build_time = time.time() - build_start
            print(f"✓ Build time: {build_time:.2f}s")

            # Query
            print(f"Querying {n_test} test points...")
            query_start = time.time()

            # Use batch query if available, otherwise loop
            if hasattr(algo, "query_batch"):
                predictions = algo.query_batch(test, k)
            else:
                predictions = [algo.query(q, k) for q in test]

            query_time = time.time() - query_start
            qps = n_test / query_time
            print(f"✓ Query time: {query_time:.2f}s ({qps:.0f} queries/sec)")

            # Compute recall
            recall = compute_recall_batch(predictions, ground_truth, k)
            print(f"✓ Recall@{k}: {recall:.4f}")

            # Get metadata for this algorithm (with prefix matching for variants)
            metadata = get_algorithm_metadata(algo_name, algo_metadata)

            # Store results
            results.append({
                "name": algo_name,
                "display": str(algo),
                "source": metadata.get("source", "Unknown"),
                "build_time": build_time,
                "query_time": query_time,
                "qps": qps,
                "recall": recall,
                "params": algo_params or {},
            })

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}\n")

    if not results:
        print("No algorithms completed successfully.")
        return

    # Sort by recall (descending)
    results.sort(key=lambda x: x["recall"], reverse=True)

    # Print results table
    print(f"{'Algorithm':<30} {'Source':<28} {'Build(s)':>10} {'QPS':>10} {'Recall@'+str(k):>12}")
    print("─" * 93)
    for r in results:
        print(f"{r['name']:<30} {r['source']:<28} {r['build_time']:>10.2f} {r['qps']:>10.0f} {r['recall']:>12.4f}")

    # Print parameter details
    print(f"\n{'─' * 80}")
    print("Algorithm Configurations")
    print(f"{'─' * 80}\n")

    for r in results:
        print(f"{r['name']}:")
        if r['params']:
            for key, value in r['params'].items():
                print(f"  {key}: {value}")
        else:
            print(f"  (default parameters)")
        print()

    print(f"{'=' * 80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ManifoldANN Benchmarking Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "config",
        nargs="?",
        default="fashion-mnist",
        help="Dataset configuration name (without .yaml)",
    )

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory for datasets",
    )

    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of neighbors to retrieve",
    )

    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List available dataset configurations",
    )

    parser.add_argument(
        "--list-algorithms",
        action="store_true",
        help="List available algorithms and their status",
    )

    args = parser.parse_args()

    # Handle --list-configs
    if args.list_configs:
        from benchmarking.utils.config import list_available_configs
        configs = list_available_configs()
        print("Available dataset configurations:")
        for config in configs:
            print(f"  - {config}")
        return

    # Handle --list-algorithms
    if args.list_algorithms:
        available = list_available_algorithms()
        print("Available algorithms:")
        for name, status in available.items():
            status_str = "✓ Available" if status else "✗ Not installed"
            print(f"  - {name:<30} {status_str}")
        return

    # Run benchmark
    run_benchmark(args.config, args.data_dir, args.k)


if __name__ == "__main__":
    main()
