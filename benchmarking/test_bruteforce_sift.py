#!/usr/bin/env python3
"""Quick test of BruteForce on SIFT dataset."""

import os
import sys
from pathlib import Path

# Configure Julia threading before any imports
if "JULIA_NUM_THREADS" not in os.environ:
    import multiprocessing
    os.environ["JULIA_NUM_THREADS"] = str(multiprocessing.cpu_count())

if "PYTHON_JULIACALL_HANDLE_SIGNALS" not in os.environ:
    os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

import time
import numpy as np

# Add benchmarking package to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmarking.utils import download_dataset, load_dataset, compute_recall_batch
from benchmarking.registry import create_algorithm

# Configuration
dataset_name = "sift-128-euclidean"
metric = "euclidean"
n_train = 10000
n_test = 1000
k = 10
data_dir = "data"

print("=" * 80)
print("BruteForce SIFT Test")
print("=" * 80)
print(f"Dataset: {dataset_name}")
print(f"Metric: {metric}")
print(f"Training points: {n_train}")
print(f"Test queries: {n_test}")
print(f"k (neighbors): {k}")

# Download and load dataset
dataset_path = download_dataset(dataset_name, data_dir)
train, test, ground_truth = load_dataset(dataset_path, n_train=n_train, n_test=n_test)

print(f"\n{'=' * 80}")
print("Testing BruteForce")
print(f"{'=' * 80}\n")

# Create BruteForce algorithm
algo = create_algorithm("MANN-BruteForce", metric, {})
print(f"Algorithm: {algo}")

# Build index
print("\nBuilding index...")
build_start = time.perf_counter()
algo.fit(train)
build_time = time.perf_counter() - build_start
print(f"✓ Build time: {build_time:.2f}s")

# Query
print(f"\nQuerying {n_test} test points...")
query_start = time.perf_counter()

if hasattr(algo, "query_batch"):
    predictions = algo.query_batch(test, k)
else:
    predictions = [algo.query(q, k) for q in test]

query_time = time.perf_counter() - query_start
qps = n_test / query_time if query_time > 0 else float("inf")
print(f"✓ Query time: {query_time:.2f}s ({qps:.0f} queries/sec)")

# Compute recall
recall = compute_recall_batch(predictions, ground_truth, k)
print(f"\n{'=' * 80}")
print(f"RESULT: Recall@{k} = {recall:.6f} ({recall*100:.4f}%)")
print(f"{'=' * 80}")

if recall == 1.0:
    print("\n✓ SUCCESS: BruteForce achieves 100% recall!")
else:
    print(f"\n✗ WARNING: BruteForce recall is {recall:.6f}, expected 1.0")
    print(f"   Missing {(1-recall)*n_test:.1f} neighbors out of {n_test*k} total")
