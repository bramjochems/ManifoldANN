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

# Threading: a `--threads` CLI flag (parsed below in `main`) is the source of
# truth at runtime, but Julia's thread count is fixed at process start by
# JULIA_NUM_THREADS, so we have to settle the env var BEFORE juliacall is
# imported. We do a coarse pre-parse here: pick up `--threads N` from argv
# (or the existing env var, or cpu_count) so JULIA_NUM_THREADS is set before
# any Julia code runs. The harness later validates that the resolved
# `--threads` value matches JULIA_NUM_THREADS and warns on mismatch.
def _early_thread_count():
    # Look for --threads N or --threads=N in argv (best-effort; argparse
    # runs again later for the full CLI).
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == "--threads" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                pass
        elif a.startswith("--threads="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                pass
    if "JULIA_NUM_THREADS" in os.environ:
        try:
            return int(os.environ["JULIA_NUM_THREADS"])
        except ValueError:
            pass
    import multiprocessing
    return multiprocessing.cpu_count()


_RESOLVED_THREADS = _early_thread_count()
if os.environ.get("JULIA_NUM_THREADS") != str(_RESOLVED_THREADS):
    if "JULIA_NUM_THREADS" in os.environ:
        # User passed an env var; if --threads disagrees, the env var wins
        # for Julia (already locked in) and we warn during run_benchmark.
        pass
    else:
        os.environ["JULIA_NUM_THREADS"] = str(_RESOLVED_THREADS)
print(f"⚙️  Threads: --threads={_RESOLVED_THREADS}, "
      f"JULIA_NUM_THREADS={os.environ.get('JULIA_NUM_THREADS', '<unset>')}")

# Suppress juliacall threading warning
if "PYTHON_JULIACALL_HANDLE_SIGNALS" not in os.environ:
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

# Now safe to import other modules
import time
import argparse
import itertools
import json
import csv
import gc
import shutil
import subprocess
import statistics
from datetime import datetime
from pathlib import Path

import numpy as np
try:
    import psutil
    _PROC = psutil.Process()
except ImportError:
    psutil = None
    _PROC = None

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

        # Header (legacy first 5 cols preserved for back-compat with
        # existing single-rep consumers; new IQR / memory cols appended).
        writer.writerow([
            "algorithm", "qps", f"recall@{k}", "build_time", "status", "error",
            "reps",
            "qps_q25", "qps_q75",
            "recall_q25", "recall_q75",
            "build_rss_delta_mb", "query_rss_delta_mb_max", "index_mb",
            "build_params", "query_params",
        ])

        for r in results:
            def _mb(b):
                return f"{b / 1e6:.1f}" if b else ""
            writer.writerow([
                r["name"],
                f"{r['qps']:.2f}",
                f"{r['recall']:.4f}",
                f"{r['build_time']:.2f}",
                "success", "",
                r.get("reps", 1),
                f"{r.get('qps_q25', r['qps']):.2f}",
                f"{r.get('qps_q75', r['qps']):.2f}",
                f"{r.get('recall_q25', r['recall']):.4f}",
                f"{r.get('recall_q75', r['recall']):.4f}",
                _mb(r.get("build_rss_delta_bytes")),
                _mb(r.get("query_rss_delta_bytes_max")),
                _mb(r.get("index_bytes")),
                json.dumps(r.get("build_params", {}), sort_keys=True),
                json.dumps(r.get("query_params", {}), sort_keys=True),
            ])

        for failed in failed_algorithms:
            writer.writerow([
                failed["name"], "N/A", "N/A", "N/A",
                "failed", failed.get("error", "Unknown error"),
                "", "", "", "", "", "", "", "", "", "",
            ])

    print(f"✓ Saved results to {csv_path}")


def save_results_json(output_dir: Path, results: list, failed_algorithms: list, k: int, reps: int):
    """Save full per-rep results to JSON for downstream analysis."""
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump({
            "k": k,
            "reps": reps,
            "results": results,
            "failed": failed_algorithms,
        }, f, indent=2, default=str)
    print(f"✓ Saved results JSON to {json_path}")


def _full_gc():
    """Force a full GC across both Python and Julia.

    Called between reps and between algorithms so a GC pause inside a
    timed window doesn't get charged to whichever algorithm was running.
    Julia's `GC.gc(true)` runs a *full* collection (incremental by
    default).
    """
    gc.collect()
    try:
        from juliacall import Main as _jl
        _jl.GC.gc()
        _jl.GC.gc(True)
    except Exception:
        pass


def _peak_rss_bytes():
    if _PROC is None:
        return None
    try:
        return int(_PROC.memory_info().rss)
    except Exception:
        return None


def _quartiles(values):
    """Return (median, q25, q75) for a list of floats. Uses linear
    interpolation; falls back gracefully on len<=2."""
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    if len(values) == 1:
        v = float(values[0])
        return (v, v, v)
    sv = sorted(float(v) for v in values)
    median = statistics.median(sv)

    def _quantile(q):
        if len(sv) == 2:
            # linear interp between the two
            return sv[0] + q * (sv[1] - sv[0])
        # numpy-equivalent linear quantile
        pos = q * (len(sv) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(sv) - 1)
        frac = pos - lo
        return sv[lo] + frac * (sv[hi] - sv[lo])

    return (median, _quantile(0.25), _quantile(0.75))


def _verify_thread_counts(threads: int):
    """Log effective thread counts for each library at run time so a
    silent oversubscription doesn't quietly skew cross-library numbers.

    Pins FAISS OMP threads and Julia BLAS threads to `threads` *before*
    probing — otherwise the banner reports the startup defaults
    (typically `nproc`) rather than what each algorithm will actually
    run with after `algo.set_num_threads(threads)` lands. Per-algorithm
    pinning still happens later; this just makes the banner honest.
    """
    print("\n--- Effective thread counts (runtime check) ---")
    # Julia
    try:
        from juliacall import Main as _jl
        jl_threads = int(_jl.seval("Threads.nthreads()"))
        print(f"  Julia Threads.nthreads()          = {jl_threads}")
        try:
            # Pin BLAS up-front so the banner reflects measurement reality.
            _jl.seval(f"using LinearAlgebra; BLAS.set_num_threads({int(threads)})")
            blas_threads = int(_jl.seval("BLAS.get_num_threads()"))
            print(f"  Julia BLAS.get_num_threads()      = {blas_threads}"
                  + ("" if blas_threads == threads
                     else f"  ⚠️  != --threads={threads}"))
        except Exception:
            pass
        if jl_threads != threads:
            print(f"  ⚠️  Julia thread count != --threads={threads} "
                  f"(JULIA_NUM_THREADS is fixed at process start)")
    except Exception as exc:
        print(f"  juliacall threading probe failed: {exc}")
    # FAISS — pin up-front so the banner matches measurement reality.
    try:
        import faiss
        try:
            faiss.omp_set_num_threads(int(threads))
        except Exception:
            pass
        faiss_t = faiss.omp_get_max_threads()
        print(f"  faiss.omp_get_max_threads()       = {faiss_t}"
              + ("" if faiss_t == threads
                 else f"  ⚠️  != --threads={threads}"))
    except Exception:
        pass
    # OpenMP / OMP_NUM_THREADS
    print(f"  OMP_NUM_THREADS                   = {os.environ.get('OMP_NUM_THREADS', '<unset>')}")
    print(f"  JULIA_NUM_THREADS                 = {os.environ.get('JULIA_NUM_THREADS', '<unset>')}")
    print()


def _emit_headtohead_csv(output_dir: Path, group_name: str, group_rows: list, k: int):
    """Emit a recall-vs-qps head-to-head CSV for a comparable group.

    `group_rows` is a list of result-dicts (already aggregated; each row
    is one algorithm × one parameter set). This is a head-to-head
    snapshot at one config per algorithm — NOT a Pareto frontier (no
    domination filtering, no parameter sweep). The general harness
    produces breadth comparisons; thesis-grade Pareto curves come from
    focused fair-compare scripts in `scripts/` (working principle).
    """
    csv_path = output_dir / f"headtohead_{group_name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "algorithm", f"recall@{k}", f"recall@{k}_q25", f"recall@{k}_q75",
            "qps", "qps_q25", "qps_q75", "build_time_s",
        ])
        for r in group_rows:
            w.writerow([
                r["name"],
                f"{r['recall']:.4f}",
                f"{r.get('recall_q25', r['recall']):.4f}",
                f"{r.get('recall_q75', r['recall']):.4f}",
                f"{r['qps']:.2f}",
                f"{r.get('qps_q25', r['qps']):.2f}",
                f"{r.get('qps_q75', r['qps']):.2f}",
                f"{r['build_time']:.2f}",
            ])
    print(f"✓ Head-to-head CSV ({group_name}): {csv_path}")


def _validate_comparable_groups(comparable_groups: dict, results: list):
    """Warn loudly if members of a comparable group don't reach
    overlapping recall ranges (indicates parameters aren't dialled to
    apples-to-apples points).

    With a single param set per algorithm we don't have a recall *range*
    per member; we use the IQR if --reps>1, otherwise the point recall.
    Signal: members differ by > 0.10 in recall.

    Skip the group if any member has a non-empty query_sweep — head-to-head
    semantics aren't defined when one side is a curve.
    """
    if not comparable_groups:
        return
    # Members with sweeps appear as multiple rows; collapse by-name picks the
    # last sweep combo silently. Detect and skip rather than mislead.
    sweep_names = {
        r["name"] for r in results
        if r.get("query_params") and r["query_params"] != {}
    }
    by_name = {r["name"]: r for r in results if r["name"] not in sweep_names}
    for group_name, body in comparable_groups.items():
        members = body.get("members", [])
        if any(m in sweep_names for m in members):
            print(f"⚠️  comparable_group '{group_name}' skipped: members use "
                  f"query_sweep (head-to-head requires a single combo per member)")
            continue
        present = [by_name[m] for m in members if m in by_name]
        missing = [m for m in members if m not in by_name]
        if missing:
            print(f"⚠️  comparable_group '{group_name}': missing members "
                  f"{missing} (skipped or failed)")
        if len(present) < 2:
            continue
        recalls = [r["recall"] for r in present]
        spread = max(recalls) - min(recalls)
        if spread > 0.10:
            print(
                f"⚠️  comparable_group '{group_name}' recall spread = {spread:.3f} "
                f"across members — parameters likely NOT apples-to-apples. "
                f"Members: " + ", ".join(
                    f"{r['name']}={r['recall']:.3f}" for r in present
                )
            )
        else:
            print(
                f"✓ comparable_group '{group_name}' recall spread = {spread:.3f} "
                f"(members within 0.10 — apples-to-apples)"
            )


def _parse_algo_entry(entry, algo_name=None):
    """Split a YAML algorithm entry into (build_combos, query_sweep).

    New form: dict with `build:` and/or `query_sweep:` keys.
    Old (flat) form: any other dict — treated entirely as build params, no
    sweep.

    Within `build:`, list-valued params expand into a Cartesian product of
    builds (one index built per combination). Scalar-valued params are
    held fixed across all combos. This lets a single YAML entry sweep
    across (M, ef_construction, ...) without duplicating the entry.

    Returns (build_combos, query_sweep) where build_combos is a list of
    dicts (each a complete build_params for one index build).

    Stray keys alongside `build`/`query_sweep` are silently ignored — warn
    so the user notices a typo before trusting the run.
    """
    if not entry:
        return [{}], {}
    if isinstance(entry, dict) and ("build" in entry or "query_sweep" in entry):
        build_raw = dict(entry.get("build") or {})
        sweep = dict(entry.get("query_sweep") or {})
        stray = set(entry) - {"build", "query_sweep"}
        if stray:
            label = f" in {algo_name}" if algo_name else ""
            print(f"⚠️  ignoring stray keys {sorted(stray)}{label} "
                  f"(use `build:` block for parameters)")
        build_combos = _expand_build_combos(build_raw)
        return build_combos, sweep
    return [dict(entry)], {}


def _expand_build_combos(build_raw):
    """Expand list-valued entries in `build:` into a cartesian product
    of fully scalar build_params dicts. Non-list values are held fixed.

    Example: {M: [8, 16], ef_construction: 200, neighbor_policy: "diversified"}
        -> [{M: 8, ef_construction: 200, neighbor_policy: "diversified"},
            {M: 16, ef_construction: 200, neighbor_policy: "diversified"}]

    Strings are NOT treated as iterables — they're scalars (e.g.
    `neighbor_policy: diversified` should not enumerate characters).
    """
    if not build_raw:
        return [{}]
    sweep_keys = []
    sweep_values = []
    fixed = {}
    for k in sorted(build_raw.keys()):
        v = build_raw[k]
        if isinstance(v, list):
            sweep_keys.append(k)
            sweep_values.append(sorted(v))
        else:
            fixed[k] = v
    if not sweep_keys:
        return [fixed]
    combos = []
    for vals in itertools.product(*sweep_values):
        combo = dict(fixed)
        combo.update(dict(zip(sweep_keys, vals)))
        combos.append(combo)
    return combos


def _sweep_combos(query_sweep):
    """Cartesian product of sweep dimensions, deterministic ordering."""
    if not query_sweep:
        return [{}]
    keys = sorted(query_sweep.keys())
    value_lists = [sorted(query_sweep[k]) for k in keys]
    combos = []
    for vals in itertools.product(*value_lists):
        combos.append(dict(zip(keys, vals)))
    return combos


def run_benchmark(config_name: str, data_dir: str = "data", k: int = 10, n_train: int = None, n_test: int = None, save_output: bool = False, threads: int = None, reps: int = 1):
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

    # Resolve thread count and warn on JULIA_NUM_THREADS mismatch.
    if threads is None:
        threads = _RESOLVED_THREADS
    julia_env = os.environ.get("JULIA_NUM_THREADS")
    if julia_env is not None and julia_env != str(threads):
        print(
            f"\n⚠️  WARNING: --threads={threads} but JULIA_NUM_THREADS={julia_env}. "
            f"Julia's thread pool is fixed at process start and CANNOT be changed at runtime; "
            f"competitor libraries WILL run with --threads={threads}. "
            f"Cross-library timings will be asymmetric.\n"
            f"   Fix: set JULIA_NUM_THREADS={threads} before invoking benchmark.py."
        )
    print(f"Threads (competitor libs): {threads}")

    # Verify Julia threading (only check once, when Julia is first initialized)
    _verify_julia_threading()

    # Effective thread-count snapshot per library, post-init. Catches the
    # silent oversubscription case (e.g. FAISS reading OMP_NUM_THREADS
    # rather than our --threads N).
    _verify_thread_counts(threads)

    # Download and load dataset
    dataset_path = download_dataset(dataset_name, data_dir)
    train, test, ground_truth = load_dataset(dataset_path, n_train=n_train, n_test=n_test)
    # Resolve `n_test` to the actual loaded count. With `n_test: 0` in YAML
    # ("use full set"), load_dataset returns the full test array but the local
    # `n_test` variable would still be 0 — and downstream `qps = n_test / time`
    # would compute 0. Same for n_train (less load-bearing, but consistent).
    n_train = int(train.shape[0])
    n_test = int(test.shape[0])

    print(f"\n{'=' * 80}")
    print("Building and Evaluating Algorithms")
    print(f"{'=' * 80}\n")

    results = []
    failed_algorithms = []

    # Iterate through configured algorithms; each algorithm entry can
    # produce multiple builds (build_sweep) and multiple queries per build
    # (query_sweep), giving (n_build_combos × n_query_combos) result rows.
    for algo_name, algo_entry in config["algorithms"].items():
        build_combos, query_sweep = _parse_algo_entry(algo_entry, algo_name)
        sweep_combos = _sweep_combos(query_sweep)

        for build_idx, build_params in enumerate(build_combos):
            print(f"\n{'─' * 80}")
            if len(build_combos) > 1:
                print(f"Algorithm: {algo_name} [build {build_idx + 1}/{len(build_combos)}]")
            else:
                print(f"Algorithm: {algo_name}")
            print(f"{'─' * 80}")
            if build_params:
                print(f"Build params: {build_params}")

            # Create algorithm instance for this specific build combo.
            try:
                algo = create_algorithm(algo_name, metric, build_params or {})
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
                # Apply thread setting (no-op for wrappers that piggy-back on
                # JULIA_NUM_THREADS).
                try:
                    algo.set_num_threads(threads)
                except Exception as exc:
                    print(f"⚠️  set_num_threads({threads}) failed: {exc}")

                # Marshal data outside the timed region so every wrapper gets
                # its preprocessing charged symmetrically (numpy -> Julia
                # matrix for Julia wrappers, dtype/contiguity coercion for
                # FAISS/hnswlib, pass-through for the rest).
                prepared_train = algo.prepare_data(train)
                try:
                    prepared_test = algo.prepare_queries(test)
                except Exception:
                    prepared_test = test

                # Build-path JIT warmup at the *actual* config (Julia
                # juliacall, numba). Costs seconds once per (algo, dim);
                # untimed.
                try:
                    algo.warmup_build(int(train.shape[1]))
                except Exception as exc:
                    print(f"⚠️  warmup_build failed: {exc}")

                # Clean GC state right before the timed build region.
                _full_gc()
                rss_before_build = _peak_rss_bytes()

                # Build index (single rep — deterministic at fixed seed).
                print("Building index...")
                build_start = time.perf_counter()
                algo.fit(prepared_train)
                build_time = time.perf_counter() - build_start
                rss_after_build = _peak_rss_bytes()
                build_rss_delta = (
                    None if (rss_before_build is None or rss_after_build is None)
                    else rss_after_build - rss_before_build
                )
                print(f"✓ Build time: {build_time:.2f}s"
                      + (f" (+{build_rss_delta/1e6:.1f} MB RSS)"
                         if build_rss_delta is not None else ""))

                # Library-reported index footprint, if exposed (depends only
                # on build, so probe once outside the sweep).
                try:
                    index_bytes = algo.memory_usage()
                except Exception:
                    index_bytes = None

                # Get metadata for this algorithm (with prefix matching for variants)
                metadata = get_algorithm_metadata(algo_name, algo_metadata)

                # Sweep over query-time parameter combinations, reusing the
                # single built index. Each combo runs its own warmup + reps.
                for combo in sweep_combos:
                    if combo:
                        print(f"\n  Query params: {combo}")
                        try:
                            algo.set_query_params(**combo)
                        except Exception as exc:
                            print(f"⚠️  set_query_params({combo}) failed: {exc}")

                    # Warm the query path at the *actual* k, batch size, and
                    # current query-param combo. Re-warm per combo so cache
                    # state is fresh for the timed reps.
                    try:
                        if hasattr(algo, "query_batch_raw"):
                            raw_warm = algo.query_batch_raw(prepared_test, k)
                            algo.finalize_batch_ids(raw_warm)
                        elif hasattr(algo, "query_batch"):
                            algo.query_batch(prepared_test, k)
                        else:
                            for q in test:
                                algo.query(q, k)
                    except Exception as exc:
                        print(f"⚠️  query warmup at config failed: {exc}")

                    # Reps: time the query path only (build is one-shot).
                    qps_samples = []
                    recall_samples = []
                    query_time_samples = []
                    rss_query_samples = []
                    predictions = None
                    for rep_i in range(reps):
                        _full_gc()
                        rss_q_before = _peak_rss_bytes()
                        print(
                            f"Querying {n_test} test points "
                            f"(rep {rep_i + 1}/{reps})..."
                        )
                        query_start = time.perf_counter()
                        if hasattr(algo, "query_batch_raw"):
                            raw = algo.query_batch_raw(prepared_test, k)
                            query_time = time.perf_counter() - query_start
                            predictions = algo.finalize_batch_ids(raw)
                        elif hasattr(algo, "query_batch"):
                            try:
                                predictions = algo.query_batch(prepared_test, k)
                            except TypeError:
                                predictions = algo.query_batch(test, k)
                            query_time = time.perf_counter() - query_start
                        else:
                            predictions = [algo.query(q, k) for q in test]
                            query_time = time.perf_counter() - query_start
                        rss_q_after = _peak_rss_bytes()
                        qps_i = n_test / query_time if query_time > 0 else float("inf")
                        recall_i = compute_recall_batch(predictions, ground_truth, k)
                        qps_samples.append(qps_i)
                        recall_samples.append(recall_i)
                        query_time_samples.append(query_time)
                        if rss_q_before is not None and rss_q_after is not None:
                            rss_query_samples.append(rss_q_after - rss_q_before)
                        print(
                            f"  rep {rep_i + 1}: query={query_time:.2f}s, "
                            f"qps={qps_i:.0f}, recall@{k}={recall_i:.4f}"
                        )

                    qps_med, qps_q25, qps_q75 = _quartiles(qps_samples)
                    recall_med, recall_q25, recall_q75 = _quartiles(recall_samples)
                    qt_med, _qt_q25, _qt_q75 = _quartiles(query_time_samples)
                    print(f"✓ Query time (median): {qt_med:.2f}s "
                          f"(qps median={qps_med:.0f} [IQR {qps_q25:.0f}–{qps_q75:.0f}])")
                    print(f"✓ Recall@{k} (median): {recall_med:.4f} "
                          f"[IQR {recall_q25:.4f}–{recall_q75:.4f}]")

                    results.append({
                        "name": algo_name,
                        "display": str(algo),
                        "source": metadata.get("source", "Unknown"),
                        "type": metadata.get("type", "unknown"),
                        "build_time": build_time,
                        "query_time": qt_med,
                        "qps": qps_med,
                        "recall": recall_med,
                        "qps_q25": qps_q25,
                        "qps_q75": qps_q75,
                        "recall_q25": recall_q25,
                        "recall_q75": recall_q75,
                        "qps_samples": qps_samples,
                        "recall_samples": recall_samples,
                        "query_time_samples": query_time_samples,
                        "reps": reps,
                        "build_rss_delta_bytes": build_rss_delta,
                        "query_rss_delta_bytes_max": (
                            max(rss_query_samples) if rss_query_samples else None
                        ),
                        "index_bytes": index_bytes,
                        "build_params": build_params or {},
                        "query_params": dict(combo),
                        # `params` kept as alias for legacy summary/print code.
                        "params": build_params or {},
                    })

                # Drop the algorithm/index reference and force a full GC so
                # the next build/algorithm starts from a clean slate.
                del algo, prepared_train, prepared_test, predictions
                _full_gc()

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
    if reps > 1:
        print(f"{'Algorithm':<30} {'Source':<24} {'Build(s)':>10} "
              f"{'QPS (med)':>10} {'QPS IQR':>16} "
              f"{'R@'+str(k):>8} {'R IQR':>14}")
        print("─" * 116)
        for r in results:
            qps_iqr = f"[{r.get('qps_q25', r['qps']):.0f}-{r.get('qps_q75', r['qps']):.0f}]"
            r_iqr = f"[{r.get('recall_q25', r['recall']):.3f}-{r.get('recall_q75', r['recall']):.3f}]"
            print(f"{r['name']:<30} {r['source']:<24} {r['build_time']:>10.2f} "
                  f"{r['qps']:>10.0f} {qps_iqr:>16} "
                  f"{r['recall']:>8.4f} {r_iqr:>14}")
    else:
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
            "reps": reps,
            "threads": threads,
        }
        save_metadata(output_dir, config_name, config, cli_args, git_info)

        # Copy config files
        copy_config_files(output_dir, config_name)

        # Save results CSV + JSON
        save_results_csv(output_dir, results, failed_algorithms, k)
        save_results_json(output_dir, results, failed_algorithms, k, reps)

        # Comparable-groups validation + head-to-head CSVs.
        comparable_groups = config.get("comparable_groups") or {}
        if comparable_groups:
            print(f"\n--- Comparable groups ---")
            _validate_comparable_groups(comparable_groups, results)
            sweep_names = {
                r["name"] for r in results
                if r.get("query_params") and r["query_params"] != {}
            }
            by_name = {r["name"]: r for r in results if r["name"] not in sweep_names}
            for group_name, body in comparable_groups.items():
                members = body.get("members", [])
                if any(m in sweep_names for m in members):
                    continue
                rows = [by_name[m] for m in members if m in by_name]
                if len(rows) >= 2:
                    _emit_headtohead_csv(output_dir, group_name, rows, k)

        print(f"\n✓ All results saved to: {output_dir}")
        print(f"{'=' * 80}\n")
    elif config.get("comparable_groups"):
        # Even without --save-output, surface the validation warning.
        print(f"\n--- Comparable groups ---")
        _validate_comparable_groups(config["comparable_groups"], results)


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

    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help=(
            "Number of query-phase repetitions; reports median + IQR of "
            "QPS and recall when reps > 1. Default 1 preserves single-shot "
            "behaviour for development sweeps. Use 3 or 5 for thesis-grade "
            "stability on noisy variants. Build phase runs once (deterministic "
            "at fixed seed)."
        ),
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "Number of threads for competitor libraries (hnswlib, FAISS, Annoy, "
            "PyNNDescent, SciPy). Default: JULIA_NUM_THREADS if set, else cpu_count(). "
            "Julia's thread pool is fixed at process start by JULIA_NUM_THREADS, so "
            "for fair comparisons set JULIA_NUM_THREADS=N to match --threads N."
        ),
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
    run_benchmark(
        args.config,
        args.data_dir,
        args.k,
        args.n_train,
        args.n_test,
        args.save_output,
        threads=args.threads,
        reps=max(1, int(args.reps)),
    )


if __name__ == "__main__":
    main()
