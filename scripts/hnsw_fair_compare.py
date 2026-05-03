#!/usr/bin/env python3
"""
Fair-comparison HNSW benchmark: ManifoldANN vs hnswlib.

Differences from benchmarking/benchmark.py:
- explicit, symmetric thread control (env JULIA_NUM_THREADS + i.set_num_threads)
- warmup actually exercises the HNSW build/query path before timing
- timed region excludes data marshalling (data already in the right layout
  on both sides) and excludes JIT/precompile cost
- two real datasets only: SIFT-128 (Euclidean) and Fashion-MNIST-784 (Euclidean)
- prints both build time and query QPS at matched recall@10

Run examples:
    JULIA_NUM_THREADS=1 uv run python scripts/hnsw_fair_compare.py --threads 1
    JULIA_NUM_THREADS=4 uv run python scripts/hnsw_fair_compare.py --threads 4
    uv run python scripts/hnsw_fair_compare.py --threads 1,4

JULIA_NUM_THREADS must be set BEFORE python imports juliacall (we set it
explicitly at the top — but it has no effect once the runtime initialised, so
either pass it via env or run separate processes for separate thread counts).
"""

import os
import sys
import time
import argparse
from pathlib import Path

# JULIA_NUM_THREADS must be set before juliacall import.
if "JULIA_NUM_THREADS" not in os.environ:
    os.environ["JULIA_NUM_THREADS"] = "1"
os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")

import numpy as np

# Make benchmarking package importable.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "benchmarking"))

from benchmarking.utils import download_dataset, load_dataset, compute_recall_batch  # noqa: E402


def warmup_julia_hnsw(jl, dim: int, dtype=np.float32):
    """Compile the HNSWIndex build + query path on small dummy data so the
    timed run measures only steady-state work. Returns nothing.
    """
    print(f"  warming up Julia HNSW path (dim={dim})...", flush=True)
    warm_data = jl.seval(f"randn(Float32, {dim}, 200)")
    warm_q = jl.seval(f"randn(Float32, {dim})")
    warm_idx = jl.build_index(jl.HNSWIndex, warm_data, M=8, ef_construction=40, ef_search=16)
    jl.query(warm_idx, warm_data, warm_q, 5)
    # Also warm a small batch query (Threads.@threads path).
    warm_qs = jl.seval(f"randn(Float32, {dim}, 8)")
    jl.query(warm_idx, warm_data, warm_qs, 5)


def run_julia(train_jl, test_jl, gt, dim, M, efc, efs, k, dtype=np.float32):
    """Time Julia HNSW build + batch query. `train_jl` and `test_jl` are
    already-converted Julia matrices (dim × n) — marshalling not timed."""
    from juliacall import Main as jl

    # Build
    t0 = time.perf_counter()
    idx = jl.build_index(
        jl.HNSWIndex, train_jl,
        M=M, ef_construction=efc, ef_search=efs,
        distance=jl.ManifoldANN.default_distance,
    )
    build_s = time.perf_counter() - t0

    # Batch query
    t0 = time.perf_counter()
    res = jl.query(idx, train_jl, test_jl, k, ef_search=efs)
    query_s = time.perf_counter() - t0

    # Convert results to a list of lists of 0-indexed ids (excluded from timed region).
    ids_jl = jl.ManifoldANN.neighbor_ids(res)
    pred = [[int(i) - 1 for i in row] for row in ids_jl]

    recall = compute_recall_batch(pred, gt, k)
    n_q = len(pred)
    return {
        "build_s": build_s,
        "query_s": query_s,
        "qps": n_q / query_s if query_s > 0 else float("inf"),
        "recall": recall,
    }


def run_hnswlib(train, test, gt, dim, M, efc, efs, k, num_threads):
    import hnswlib

    n = train.shape[0]

    # Warmup: build a small index + tiny query to compile any lazy paths.
    # hnswlib is C++ so no JIT, but populates allocator pools / caches. Cheap
    # to do for parity with the Julia warmup.
    w = hnswlib.Index(space="l2", dim=dim)
    w.init_index(max_elements=200, ef_construction=40, M=8)
    w.set_num_threads(num_threads)
    w.add_items(train[:200].astype(np.float32, copy=False))
    w.set_ef(16)
    w.knn_query(test[:8].astype(np.float32, copy=False), k=5)

    # Real run.
    train_f32 = np.ascontiguousarray(train, dtype=np.float32)
    test_f32 = np.ascontiguousarray(test, dtype=np.float32)
    ids_arr = np.arange(n, dtype=np.int64)

    t0 = time.perf_counter()
    idx = hnswlib.Index(space="l2", dim=dim)
    idx.init_index(max_elements=n, ef_construction=efc, M=M)
    idx.set_num_threads(num_threads)
    idx.add_items(train_f32, ids_arr)
    build_s = time.perf_counter() - t0

    idx.set_ef(efs)
    idx.set_num_threads(num_threads)

    t0 = time.perf_counter()
    labels, _dists = idx.knn_query(test_f32, k=k)
    query_s = time.perf_counter() - t0

    pred = labels.tolist()
    recall = compute_recall_batch(pred, gt, k)
    return {
        "build_s": build_s,
        "query_s": query_s,
        "qps": test.shape[0] / query_s if query_s > 0 else float("inf"),
        "recall": recall,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1",
                    help="Comma-separated thread counts to compare. NOTE: Julia's "
                         "thread count is fixed at process start by JULIA_NUM_THREADS; "
                         "this script's hnswlib side respects --threads dynamically. "
                         "For multi-thread Julia, run this script multiple times "
                         "with JULIA_NUM_THREADS set externally.")
    ap.add_argument("--dataset", default="sift,fashion-mnist",
                    help="Comma-separated dataset names (sift, fashion-mnist)")
    ap.add_argument("--n-train", type=int, default=10_000)
    ap.add_argument("--n-test", type=int, default=1_000)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=80)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--no-julia", action="store_true",
                    help="Skip Julia (useful for sanity-checking hnswlib timings only)")
    args = ap.parse_args()

    thread_counts = [int(x) for x in args.threads.split(",")]
    datasets = [x.strip() for x in args.dataset.split(",")]

    print(f"JULIA_NUM_THREADS = {os.environ.get('JULIA_NUM_THREADS')}")
    print(f"hnswlib --threads sweep: {thread_counts}")
    print(f"datasets: {datasets}")
    print(f"n_train={args.n_train} n_test={args.n_test} k={args.k} "
          f"M={args.M} ef_c={args.ef_construction} ef_s={args.ef_search}\n")

    DATASET_NAMES = {
        "sift": "sift-128-euclidean",
        "fashion-mnist": "fashion-mnist-784-euclidean",
    }

    # Lazy-init Julia after env vars are set.
    jl = None
    if not args.no_julia:
        from juliacall import Main as jl_mod
        jl = jl_mod
        jl.seval(f'using Pkg; Pkg.activate("{REPO}")')
        jl.seval("using ManifoldANN")
        # We need a per-dim warmup. Cache by dim.
        warmed_dims = set()

    data_dir = REPO / "benchmarking" / "data"

    rows = []
    for dskey in datasets:
        ds = DATASET_NAMES[dskey]
        path = download_dataset(ds, str(data_dir))
        train, test, gt = load_dataset(path, n_train=args.n_train, n_test=args.n_test)
        n, dim = train.shape
        print(f"\n=== {dskey}: n={n} d={dim} ===")

        # Pre-marshal Julia data ONCE, outside the timed region.
        if jl is not None:
            if dim not in warmed_dims:
                warmup_julia_hnsw(jl, dim)
                warmed_dims.add(dim)
            to_mat = jl.seval("x -> Matrix{Float32}(x)")
            train_jl = to_mat(np.asfortranarray(train.T, dtype=np.float32))
            test_jl  = to_mat(np.asfortranarray(test.T,  dtype=np.float32))

            r = run_julia(train_jl, test_jl, gt, dim,
                          args.M, args.ef_construction, args.ef_search, args.k)
            print(f"  Julia HNSW (JULIA_NUM_THREADS={os.environ['JULIA_NUM_THREADS']}): "
                  f"build={r['build_s']:.3f}s qps={r['qps']:.0f} recall@{args.k}={r['recall']:.4f}")
            rows.append((dskey, "ManifoldANN-HNSW",
                         os.environ['JULIA_NUM_THREADS'], r))

        for nt in thread_counts:
            r = run_hnswlib(train, test, gt, dim,
                            args.M, args.ef_construction, args.ef_search, args.k, nt)
            print(f"  hnswlib  (threads={nt}): "
                  f"build={r['build_s']:.3f}s qps={r['qps']:.0f} recall@{args.k}={r['recall']:.4f}")
            rows.append((dskey, "hnswlib", str(nt), r))

    # Summary
    print("\n=== summary ===")
    print(f"{'dataset':<16} {'lib':<18} {'threads':<8} {'build_s':>9} {'qps':>9} {'recall':>8}")
    for ds, lib, t, r in rows:
        print(f"{ds:<16} {lib:<18} {t:<8} {r['build_s']:>9.3f} {r['qps']:>9.0f} {r['recall']:>8.4f}")


if __name__ == "__main__":
    main()
