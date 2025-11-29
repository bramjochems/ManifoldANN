#!/usr/bin/env python3
"""
ManifoldANN Benchmarking Suite.

Compare ManifoldANN algorithms against other popular ANN libraries
using standardized datasets and metrics.
"""

# CRITICAL: Configure Julia threading BEFORE any imports
# Julia threading must be set before the Julia runtime initializes, which happens
# on the first juliacall import. We do this at the very top of the file.
import os
import sys

if "JULIA_NUM_THREADS" not in os.environ:
    import multiprocessing
    n_threads = multiprocessing.cpu_count()
    os.environ["JULIA_NUM_THREADS"] = str(n_threads)
    print(f"⚙️  Auto-configured Julia threading: JULIA_NUM_THREADS={n_threads}")
else:
    print(f"⚙️  Using existing Julia threading: JULIA_NUM_THREADS={os.environ['JULIA_NUM_THREADS']}")

# Suppress juliacall threading warning
if "PYTHON_JULIACALL_HANDLE_SIGNALS" not in os.environ:
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

# Now safe to import other modules
import time
import argparse
import json
import csv
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

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


# Cache to avoid checking threading multiple times
_julia_threading_verified = False

def _verify_julia_threading():
    """Verify that Julia threading is properly configured.

    This checks the actual number of threads Julia is using and warns if it's
    suboptimal. Only runs once per session.
    """
    global _julia_threading_verified

    if _julia_threading_verified:
        return

    try:
        from juliacall import Main as jl

        # Check actual thread count in Julia
        n_threads = jl.seval("Threads.nthreads()")

        if n_threads == 1:
            print(f"\n⚠️  WARNING: Julia is using only 1 thread!")
            print(f"   Multi-threaded algorithms (IVF-HNSW, etc.) will run sequentially.")
            print(f"   To enable threading, set JULIA_NUM_THREADS before running:")
            print(f"   export JULIA_NUM_THREADS=$(nproc)  # Linux/WSL")
            print(f"   export JULIA_NUM_THREADS=$(sysctl -n hw.ncpu)  # macOS")
            print()
        else:
            print(f"✓ Julia threading enabled: {n_threads} threads\n")

        _julia_threading_verified = True

    except Exception as e:
        # If juliacall isn't available yet, skip verification
        # (it will be checked when the first Julia algorithm runs)
        pass


def get_git_info():
    """Get current git commit SHA and dirty state.

    Returns:
        dict with 'sha' and 'dirty' keys, or None if not in git repo
    """
    try:
        # Get current commit SHA
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Check if repo is dirty
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = len(status) > 0

        return {"sha": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def create_output_directory(base_dir: str = None):
    """Create timestamped output directory.

    Args:
        base_dir: Base directory for outputs (defaults to output/ relative to this script)

    Returns:
        Path object for the created directory
    """
    if base_dir is None:
        # Use output directory relative to this script's location
        script_dir = Path(__file__).parent
        base_dir = script_dir / "output"
    else:
        base_dir = Path(base_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = base_dir / f"results_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_metadata(output_dir: Path, config_name: str, config: dict,
                  cli_args: dict, git_info: dict):
    """Save metadata JSON file with run information.

    Args:
        output_dir: Output directory path
        config_name: Name of the config used
        config: Configuration dictionary
        cli_args: Command-line arguments
        git_info: Git information (SHA and dirty state)
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "git": git_info,
        "cli_arguments": cli_args,
        "dataset_config": {
            "config_name": config_name,
            "dataset": config.get("dataset"),
            "metric": config.get("metric"),
            "n_train": config.get("n_train"),
            "n_test": config.get("n_test"),
        }
    }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata to {metadata_path}")


def copy_config_files(output_dir: Path, config_name: str):
    """Copy configuration files to output directory.

    Args:
        output_dir: Output directory path
        config_name: Name of the dataset config used
    """
    # Use configs directory relative to this script's location
    script_dir = Path(__file__).parent
    config_dir = script_dir / "configs"

    # Always copy algorithms.yaml
    algorithms_yaml = config_dir / "algorithms.yaml"
    if algorithms_yaml.exists():
        shutil.copy(algorithms_yaml, output_dir / "algorithms.yaml")
        print(f"✓ Copied algorithms.yaml")

    # Copy dataset config
    dataset_config = config_dir / f"{config_name}.yaml"
    if dataset_config.exists():
        shutil.copy(dataset_config, output_dir / f"{config_name}.yaml")
        print(f"✓ Copied {config_name}.yaml")


def save_results_csv(output_dir: Path, results: list, failed_algorithms: list, k: int):
    """Save results to CSV file.

    Args:
        output_dir: Output directory path
        results: List of successful result dictionaries
        failed_algorithms: List of failed algorithm info
        k: Number of neighbors used
    """
    csv_path = output_dir / "results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(["algorithm", "qps", f"recall@{k}", "build_time", "status", "error"])

        # Write successful results
        for r in results:
            writer.writerow([
                r["name"],
                f"{r['qps']:.2f}",
                f"{r['recall']:.4f}",
                f"{r['build_time']:.2f}",
                "success",
                ""
            ])

        # Write failed algorithms
        for failed in failed_algorithms:
            writer.writerow([
                failed["name"],
                "N/A",
                "N/A",
                "N/A",
                "failed",
                failed.get("error", "Unknown error")
            ])

    print(f"✓ Saved results to {csv_path}")


def run_benchmark(config_name: str, data_dir: str = "data", k: int = 10, n_train: int = None, n_test: int = None, save_output: bool = False):
    """Run benchmarks for a single dataset configuration.

    Args:
        config_name: Name of the config file (without .yaml extension)
        data_dir: Directory for downloading/storing datasets
        k: Number of neighbors to retrieve
        n_train: Number of training points (None uses config, 0 uses full dataset)
        n_test: Number of test queries (None uses config, 0 uses full dataset)
        save_output: If True, save results to timestamped directory under benchmarking/output/
    """
    # Load configuration and algorithm metadata
    print("=" * 80)
    print(f"Loading configuration: {config_name}")
    print("=" * 80)
    config = load_config(config_name)
    algo_metadata = load_algorithm_metadata()

    # Setup output directory if requested
    output_dir = None
    if save_output:
        output_dir = create_output_directory()
        print(f"\n📁 Output directory: {output_dir}")

        # Get git info
        git_info = get_git_info()
        if git_info:
            dirty_str = " (dirty)" if git_info["dirty"] else ""
            print(f"📝 Git commit: {git_info['sha'][:8]}{dirty_str}")
        else:
            print("⚠️  Not in a git repository")
            git_info = {"sha": None, "dirty": None}

    dataset_name = config["dataset"]
    metric = config["metric"]

    # Use command-line values if provided, otherwise use config values
    n_train = n_train if n_train is not None else config["n_train"]
    n_test = n_test if n_test is not None else config["n_test"]

    print(f"Dataset: {dataset_name}")
    print(f"Metric: {metric}")
    print(f"Training points: {n_train}")
    print(f"Test queries: {n_test}")
    print(f"k (neighbors): {k}")

    # Verify Julia threading (only check once, when Julia is first initialized)
    _verify_julia_threading()

    # Download and load dataset
    dataset_path = download_dataset(dataset_name, data_dir)
    train, test, ground_truth = load_dataset(dataset_path, n_train=n_train, n_test=n_test)

    print(f"\n{'=' * 80}")
    print("Building and Evaluating Algorithms")
    print(f"{'=' * 80}\n")

    results = []
    failed_algorithms = []

    # Iterate through configured algorithms
    for algo_name, algo_params in config["algorithms"].items():
        print(f"\n{'─' * 80}")
        print(f"Algorithm: {algo_name}")
        print(f"{'─' * 80}")

        # Create algorithm instance
        try:
            algo = create_algorithm(algo_name, metric, algo_params or {})
        except ValueError as e:
            print(f"✗ Error: {e}")
            failed_algorithms.append({
                "name": algo_name,
                "error": str(e)
            })
            continue

        if algo is None:
            print(f"⚠️  Skipped (library not available)")
            failed_algorithms.append({
                "name": algo_name,
                "error": "Library not available"
            })
            continue

        print(f"Configuration: {algo}")

        try:
            # Build index
            print("Building index...")
            build_start = time.perf_counter()
            algo.fit(train)
            build_time = time.perf_counter() - build_start
            print(f"✓ Build time: {build_time:.2f}s")

            # Query
            print(f"Querying {n_test} test points...")
            query_start = time.perf_counter()

            # Use batch query if available, otherwise loop
            if hasattr(algo, "query_batch"):
                predictions = algo.query_batch(test, k)
            else:
                predictions = [algo.query(q, k) for q in test]

            query_time = time.perf_counter() - query_start
            qps = n_test / query_time if query_time > 0 else float("inf")
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
                "type": metadata.get("type", "unknown"),
                "build_time": build_time,
                "query_time": query_time,
                "qps": qps,
                "recall": recall,
                "params": algo_params or {},
            })

        except Exception as e:
            error_msg = str(e)
            print(f"✗ Error: {error_msg}")
            import traceback
            traceback.print_exc()

            # Track failed algorithm
            failed_algorithms.append({
                "name": algo_name,
                "error": error_msg
            })
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
    print(f"{'Algorithm':<30} {'Source':<28} {'Type':<10} {'Build(s)':>10} {'QPS':>10} {'Recall@'+str(k):>12}")
    print("─" * 103)
    for r in results:
        print(f"{r['name']:<30} {r['source']:<28} {r['type']:<10} {r['build_time']:>10.2f} {r['qps']:>10.0f} {r['recall']:>12.4f}")

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

    # Save outputs if requested
    if save_output and output_dir:
        print(f"{'=' * 80}")
        print("Saving Results")
        print(f"{'=' * 80}\n")

        # Save metadata
        cli_args = {
            "config": config_name,
            "data_dir": data_dir,
            "k": k,
            "n_train": n_train,
            "n_test": n_test,
        }
        save_metadata(output_dir, config_name, config, cli_args, git_info)

        # Copy config files
        copy_config_files(output_dir, config_name)

        # Save results CSV
        save_results_csv(output_dir, results, failed_algorithms, k)

        print(f"\n✓ All results saved to: {output_dir}")
        print(f"{'=' * 80}\n")


def main():
    """Main entry point."""
    # Get available configs for help text
    from benchmarking.utils.config import list_available_configs
    available_configs = list_available_configs()
    config_list = ", ".join(available_configs) if available_configs else "none found"

    parser = argparse.ArgumentParser(
        description="ManifoldANN Benchmarking Suite - Compare ANN algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python benchmark.py fashion-mnist          # Run fashion-mnist benchmark
  python benchmark.py sift -k 20             # Run SIFT with k=20 neighbors
  python benchmark.py mnist --n-train 5000   # Use 5000 training points
  python benchmark.py mnist --n-train 0      # Use full training dataset
  python benchmark.py --list-configs         # Show all available datasets
  python benchmark.py --list-algorithms      # Show all available algorithms

Available datasets:
  {config_list}
        """,
    )

    parser.add_argument(
        "config",
        nargs="?",
        default="fashion-mnist",
        help="Dataset configuration name (without .yaml extension)",
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
        "--n-train",
        type=int,
        default=None,
        help="Number of training points (0 or omit for config default, use full dataset with large value)",
    )

    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="Number of test queries (0 or omit for config default, use full dataset with large value)",
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

    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save results to timestamped directory under benchmarking/output/",
    )

    args = parser.parse_args()

    # Handle --list-configs
    if args.list_configs:
        configs = available_configs
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
    run_benchmark(args.config, args.data_dir, args.k, args.n_train, args.n_test, args.save_output)


if __name__ == "__main__":
    main()
