#!/usr/bin/env python3
"""
Comprehensive ORC (Ollivier-Ricci Curvature) Benchmark

Compares curvature computation across:
- ManifoldANN (Julia) with different solvers
- GraphRicciCurvature (Python)
- orcml (Python reference implementation)

Usage:
    python benchmark_orc.py [--sizes 100,500,1000] [--k 15] [--runs 3]
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Required dependencies
try:
    from juliacall import Main as jl
except ImportError:
    print("Error: juliacall not found. Install with: pip install juliacall")
    sys.exit(1)

# Configure Julia signal handling
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

# Optional dependencies
try:
    import networkx as nx
    from GraphRicciCurvature.OllivierRicci import OllivierRicci
    GRAPH_RICCI_AVAILABLE = True
except ImportError:
    GRAPH_RICCI_AVAILABLE = False
    print("Warning: GraphRicciCurvature not available. Install with: pip install GraphRicciCurvature")

# Check for orcml
ORCML_AVAILABLE = False
try:
    # Try common locations
    for path in ['/tmp/orcml', str(Path.home() / 'orcml'), '../orcml']:
        orcml_path = Path(path)
        if orcml_path.exists():
            sys.path.insert(0, str(orcml_path))
            from src.utils.graph_utils import build_knn_graph, compute_eff_eps
            from src.ollivier_ricci import OllivierRicciCurvature as ORCMLCurvature
            ORCML_AVAILABLE = True
            print(f"✓ orcml found at {orcml_path}")
            break
except ImportError:
    pass

if not ORCML_AVAILABLE:
    print("Warning: orcml not available. Clone from: https://github.com/TristanSaidi/orcml")


def setup_julia():
    """Initialize Julia and load ManifoldANN"""
    print("Initializing Julia...")
    jl.seval('ENV["JULIA_NUM_THREADS"] = string(Sys.CPU_THREADS)')
    jl.seval('using Pkg')

    # Activate the root ManifoldANN project (parent of benchmarking/)
    root_dir = Path(__file__).parent.parent
    jl.seval(f'Pkg.activate("{root_dir}")')

    jl.seval('using ManifoldANN')
    jl.seval('using LinearAlgebra')
    jl.seval('using Random')
    jl.seval('using Statistics')

    # JIT warmup
    print("Warming up Julia JIT...")
    jl.seval('Random.seed!(42)')
    jl.seval('warmup_data = randn(Float64, 3, 10)')
    jl.seval('warmup_index = ManifoldANN.build_index(ManifoldANN.BruteForceIndex, warmup_data)')
    jl.seval('warmup_graph = ManifoldANN.build_knn_graph(warmup_index, warmup_data; k=4, directed=false)')
    jl.seval('ManifoldANN.compute_all_curvatures(warmup_graph, warmup_data; solver=ManifoldANN.HungarianSolver(), fallback_solver=ManifoldANN.NetworkSimplexSolver())')
    print("✓ Julia initialized")


def generate_test_data(n, dim, k, seed=42):
    """Generate random test data and k-NN graph"""
    jl.seval(f'Random.seed!({seed})')
    jl.seval(f'test_data = randn(Float64, {dim}, {n})')
    jl.seval(f'test_index = ManifoldANN.build_index(ManifoldANN.BruteForceIndex, test_data)')

    # Build both directed and undirected graphs
    jl.seval(f'test_graph_directed = ManifoldANN.build_knn_graph(test_index, test_data; k={k}, directed=true)')
    jl.seval(f'test_graph_undirected = ManifoldANN.build_knn_graph(test_index, test_data; k={k}, directed=false)')

    # Get data as numpy array
    data = np.array(jl.test_data).T  # Convert to n × d
    return data


def benchmark_julia_orc(n, k, solver_name, solver_code, config_name, graph_directed, cost_metric, denom_metric, exclude_endpoints):
    """Benchmark Julia ManifoldANN ORC computation"""

    # Select graph based on directedness required
    graph_var = 'test_graph_directed' if graph_directed else 'test_graph_undirected'

    # Run benchmark
    start = time.perf_counter()
    jl.seval(f'''
        results = ManifoldANN.compute_all_curvatures(
            {graph_var}, test_data;
            cost_metric=:{cost_metric},
            denominator_metric=:{denom_metric},
            exclude_edge_endpoints={str(exclude_endpoints).lower()},
            solver={solver_code},
            fallback_solver=ManifoldANN.NetworkSimplexSolver()
        )
    ''')
    elapsed = time.perf_counter() - start

    # Get edge count and curvature statistics
    n_edges = int(jl.seval('length(results)'))

    # Extract curvatures and compute statistics
    jl.seval('curvatures = [result.curvature for result in values(results)]')
    n_edges_pruned = int(jl.seval('sum(curvatures .< 0)'))
    mean_curvature = float(jl.seval('mean(curvatures)'))

    return {
        'method': config_name,
        'solver': solver_name,
        'source': 'ManifoldANN',
        'n_edges': n_edges,
        'n_edges_pruned': n_edges_pruned,
        'mean_curvature': mean_curvature,
        'time_sec': elapsed
    }


def benchmark_graphricci(data, k):
    """Benchmark Python GraphRicciCurvature"""
    if not GRAPH_RICCI_AVAILABLE:
        return None

    from sklearn.neighbors import kneighbors_graph

    # Build k-NN graph
    knn = kneighbors_graph(data, k, mode='distance', include_self=False)

    results = []

    # Compute ORC with Sinkhorn (approximate)
    G_sinkhorn = nx.from_scipy_sparse_array(knn, create_using=nx.DiGraph())
    start = time.perf_counter()
    orc_sinkhorn = OllivierRicci(G_sinkhorn, alpha=0.5, method="Sinkhorn", verbose="ERROR")
    orc_sinkhorn.compute_ricci_curvature()
    elapsed_sinkhorn = time.perf_counter() - start

    # Extract curvatures
    curvatures_sinkhorn = [orc_sinkhorn.G[u][v]['ricciCurvature'] for u, v in orc_sinkhorn.G.edges()]
    n_edges_sinkhorn = len(curvatures_sinkhorn)
    n_edges_pruned_sinkhorn = sum(1 for c in curvatures_sinkhorn if c < 0)
    mean_curvature_sinkhorn = np.mean(curvatures_sinkhorn) if curvatures_sinkhorn else 0.0

    results.append({
        'method': 'ORC',
        'solver': 'Sinkhorn',
        'source': 'GraphRicciCurvature',
        'n_edges': n_edges_sinkhorn,
        'n_edges_pruned': n_edges_pruned_sinkhorn,
        'mean_curvature': mean_curvature_sinkhorn,
        'time_sec': elapsed_sinkhorn
    })

    # Compute ORC with OTD (exact - uses network simplex)
    # Label as NetworkSimplex in output for clarity
    try:
        G_otd = nx.from_scipy_sparse_array(knn, create_using=nx.DiGraph())
        start = time.perf_counter()
        orc_otd = OllivierRicci(G_otd, alpha=0.5, method="OTD", verbose="ERROR")
        orc_otd.compute_ricci_curvature()
        elapsed_otd = time.perf_counter() - start

        # Extract curvatures
        curvatures_otd = [orc_otd.G[u][v]['ricciCurvature'] for u, v in orc_otd.G.edges()]
        n_edges_otd = len(curvatures_otd)
        n_edges_pruned_otd = sum(1 for c in curvatures_otd if c < 0)
        mean_curvature_otd = np.mean(curvatures_otd) if curvatures_otd else 0.0

        results.append({
            'method': 'ORC',
            'solver': 'NetworkSimplex',  # Label as NetworkSimplex (OTD uses network simplex)
            'source': 'GraphRicciCurvature',
            'n_edges': n_edges_otd,
            'n_edges_pruned': n_edges_pruned_otd,
            'mean_curvature': mean_curvature_otd,
            'time_sec': elapsed_otd
        })
    except Exception as e:
        print(f"      (OTD method not available: {e})")

    return results


def benchmark_orcml(data, k):
    """Benchmark orcml reference implementation"""
    if not ORCML_AVAILABLE:
        return None

    # Build k-NN graph (orcml uses undirected)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
    distances, indices = nbrs.kneighbors(data)

    # Remove self-loops
    indices = indices[:, 1:]
    distances = distances[:, 1:]

    # Build graph structure for orcml
    G = {}
    for i in range(len(data)):
        G[i] = list(indices[i])

    # Compute effective epsilon
    eff_eps = compute_eff_eps(data, indices, k)

    # Compute ORC
    start = time.perf_counter()
    orc = ORCMLCurvature(G, data, eff_eps, exclude_edge_endpoints=True)
    curvatures = orc.compute_ricci_curvature()
    elapsed = time.perf_counter() - start

    # Extract curvature values
    curvature_values = list(curvatures.values())
    n_edges = len(curvature_values)
    n_edges_pruned = sum(1 for c in curvature_values if c < 0)
    mean_curvature = np.mean(curvature_values) if curvature_values else 0.0

    return {
        'method': 'ORC-orcml',
        'solver': 'Hungarian',
        'source': 'orcml',
        'n_edges': n_edges,
        'n_edges_pruned': n_edges_pruned,
        'mean_curvature': mean_curvature,
        'time_sec': elapsed
    }


def run_benchmarks(sizes, k, dim, runs, output_dir):
    """Run complete benchmark suite"""

    # Setup Julia
    setup_julia()

    # Results storage
    all_results = []

    # Julia solver configurations
    julia_solvers = [
        ('Hungarian', 'ManifoldANN.HungarianSolver()'),
        ('NetworkSimplex', 'ManifoldANN.NetworkSimplexSolver()'),
        ('LPReference', 'ManifoldANN.LPReferenceSolver()'),
        ('Sinkhorn', 'ManifoldANN.SinkhornSolver(reg=0.1)'),
    ]

    # ORC configurations
    orc_configs = [
        ('ORC', True, 'euclidean', 'euclidean', False),  # Default: directed, no exclusion
        ('ORC-orcml', False, 'geodesic_normalized', 'normalized', True),  # orcml: undirected, exclusion
    ]

    for n in sizes:
        print(f"\n{'='*80}")
        print(f"Benchmarking: n={n}, k={k}, dim={dim}")
        print(f"{'='*80}")

        for run in range(runs):
            print(f"\n  Run {run+1}/{runs}")

            # Generate test data
            seed = 42 + run
            data = generate_test_data(n, dim, k, seed)

            # Benchmark Julia with different configurations
            for config_name, directed, cost_metric, denom_metric, exclude_endpoints in orc_configs:
                for solver_name, solver_code in julia_solvers:
                    try:
                        result = benchmark_julia_orc(
                            n, k, solver_name, solver_code,
                            config_name, directed, cost_metric, denom_metric, exclude_endpoints
                        )
                        result.update({'n': n, 'k': k, 'dim': dim, 'run': run})
                        all_results.append(result)
                        print(f"    ✓ {config_name:12s} {solver_name:15s} {result['time_sec']:.3f}s")
                    except Exception as e:
                        print(f"    ✗ {config_name:12s} {solver_name:15s} ERROR: {e}")

            # Benchmark GraphRicciCurvature (Python)
            if GRAPH_RICCI_AVAILABLE:
                try:
                    grc_results = benchmark_graphricci(data, k)
                    if grc_results:
                        for result in grc_results:
                            result.update({'n': n, 'k': k, 'dim': dim, 'run': run})
                            all_results.append(result)
                            print(f"    ✓ GraphRicciCurvature {result['solver']:10s} {result['time_sec']:.3f}s")
                except Exception as e:
                    print(f"    ✗ GraphRicciCurvature ERROR: {e}")

            # Benchmark orcml
            if ORCML_AVAILABLE:
                try:
                    result = benchmark_orcml(data, k)
                    if result:
                        result.update({'n': n, 'k': k, 'dim': dim, 'run': run})
                        all_results.append(result)
                        print(f"    ✓ orcml               {result['time_sec']:.3f}s")
                except Exception as e:
                    print(f"    ✗ orcml ERROR: {e}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"orc_benchmark_{timestamp}.csv"

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'n', 'k', 'dim', 'run', 'method', 'solver', 'source',
            'n_edges', 'n_edges_pruned', 'mean_curvature', 'time_sec'
        ])
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Comprehensive ORC Benchmark')
    parser.add_argument('--sizes', type=str, default='1000',
                       help='Comma-separated graph sizes (default: 1000)')
    parser.add_argument('--k', type=int, default=20,
                       help='Number of neighbors (default: 20)')
    parser.add_argument('--dim', type=int, default=50,
                       help='Data dimensionality (default: 50)')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per configuration (default: 3)')
    parser.add_argument('--output-dir', type=str, default='results/orc',
                       help='Output directory (default: results/orc)')

    args = parser.parse_args()

    # Parse sizes
    sizes = [int(s.strip()) for s in args.sizes.split(',')]

    # Create output directory
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ORC CURVATURE BENCHMARK")
    print("="*80)
    print(f"Graph sizes: {sizes}")
    print(f"k (neighbors): {args.k}")
    print(f"Dimensionality: {args.dim}")
    print(f"Runs per config: {args.runs}")
    print(f"Output directory: {output_dir}")
    print(f"\nDependencies:")
    print(f"  ManifoldANN (Julia): ✓")
    print(f"  GraphRicciCurvature: {'✓' if GRAPH_RICCI_AVAILABLE else '✗'}")
    print(f"  orcml: {'✓' if ORCML_AVAILABLE else '✗'}")
    print("="*80)

    # Run benchmarks
    run_benchmarks(sizes, args.k, args.dim, args.runs, output_dir)


if __name__ == '__main__':
    main()
