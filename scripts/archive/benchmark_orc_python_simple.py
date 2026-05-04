"""
Simple Python ORC benchmark using pre-generated data from Julia.

This version:
1. Reads the same test data Julia used
2. Only benchmarks orcml (if available)
3. Simpler setup - fewer dependencies
"""

import numpy as np
import time
import sys
from pathlib import Path

# Check for numpy
try:
    from sklearn.neighbors import kneighbors_graph
    import networkx as nx
except ImportError as e:
    print(f"Error: Missing dependencies")
    print(f"{e}")
    print("Install with: pip install scikit-learn networkx")
    sys.exit(1)

# Check for orcml
orcml_available = False
try:
    sys.path.insert(0, '/tmp/orcml')
    from src.utils.graph_utils import build_knn_graph, compute_eff_eps
    from src.ollivier_ricci import OllivierRicciCurvature
    orcml_available = True
    print("✓ orcml found at /tmp/orcml")
except ImportError as e:
    print(f"✗ orcml not available: {e}")
    print("This benchmark requires orcml")
    sys.exit(1)

print("="*80)
print("Python ORC Benchmark (orcml only)")
print("="*80)

# Use the same data Julia generated
SIZES = [100, 500, 1000]
K = 15
results = []

for n in SIZES:
    data_file = f"benchmark_results/benchmark_data_n{n}.csv"

    # Check if Julia generated this file
    if not Path(data_file).exists():
        print(f"\n✗ Data file not found: {data_file}")
        print(f"  Generate with Julia first!")
        continue

    print(f"\n{'='*80}")
    print(f"Graph Size: n={n}")
    print(f"{'='*80}")

    # Load data (Julia saves as d x n, we need n x d)
    data = np.loadtxt(data_file, delimiter=',')
    print(f"Loaded data: {data.shape}")

    try:
        print(f"\nBuilding k-NN graph (k={K})...")
        start_build = time.time()

        # Build graph using orcml's function (matches our Julia implementation)
        G = build_knn_graph(
            data,
            n_neighbors=K,
            mode='distance',
            include_self=False,
            symmetrize=True  # Undirected
        )

        build_time = time.time() - start_build

        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"  Build time: {build_time:.3f}s")

        print(f"\nComputing ORC curvatures...")
        start_orc = time.time()

        # Compute ORC (orcml configuration)
        orc_computer = OllivierRicciCurvature(
            G,
            exclude_edge_endpoints=True,
            weight="effective_eps"
        )
        orc_computer.compute_ricci_curvature()

        orc_time = time.time() - start_orc
        total_time = build_time + orc_time

        # Extract curvatures
        curvatures = []
        for u, v, data in G.edges(data=True):
            if 'ricciCurvature' in data:
                curvatures.append(data['ricciCurvature'])

        n_edges = len(curvatures)
        mean_curv = np.mean(curvatures)
        std_curv = np.std(curvatures)
        ms_per_edge = orc_time / n_edges * 1000

        print(f"  ✓ {orc_time:.3f}s ({ms_per_edge:.3f}ms/edge)")
        print(f"    Mean κ: {mean_curv:.4f}")
        print(f"    Std κ: {std_curv:.4f}")

        results.append({
            'n': n,
            'build_time_sec': build_time,
            'orc_time_sec': orc_time,
            'total_time_sec': total_time,
            'n_edges': n_edges,
            'ms_per_edge': ms_per_edge,
            'mean_curv': mean_curv,
            'std_curv': std_curv
        })

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

# Save results
if results:
    print(f"\n{'='*80}")
    print("Saving Results")
    print(f"{'='*80}")

    import csv
    with open('benchmark_results/orc_benchmark_python_orcml.csv', 'w', newline='') as f:
        fieldnames = ['n', 'build_time_sec', 'orc_time_sec', 'total_time_sec',
                      'n_edges', 'ms_per_edge', 'mean_curv', 'std_curv']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Saved to benchmark_results/orc_benchmark_python_orcml.csv")

    # Print summary table
    print(f"\n{'='*80}")
    print("Performance Summary")
    print(f"{'='*80}\n")
    print(f"{'n':>6} | {'Edges':>7} | {'Total(s)':>9} | {'ORC(s)':>8} | {'ms/edge':>8} | {'Mean κ':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['n']:6d} | {r['n_edges']:7d} | {r['total_time_sec']:9.3f} | "
              f"{r['orc_time_sec']:8.3f} | {r['ms_per_edge']:8.3f} | {r['mean_curv']:8.4f}")

print(f"\n{'='*80}")
print("Python Benchmark Complete!")
print(f"{'='*80}")
