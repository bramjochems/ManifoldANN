#!/usr/bin/env python3
"""
Extract Edge-Level Curvatures for Distribution Analysis

This script generates edge curvatures for visualization and distribution analysis.
Unlike benchmark_orc.py which compares solvers, this focuses on extracting
curvature distributions for a single solver configuration.

Usage:
    python extract_edge_curvatures.py --sizes 100,200,400 --k 10 --runs 5 --solver hungarian
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
    jl.seval('ManifoldANN.compute_all_curvatures(warmup_graph, warmup_data; solver=ManifoldANN.HungarianSolver())')
    print("✓ Julia initialized")


def generate_test_data(n, dim, k, seed=42):
    """Generate random test data and k-NN graph"""
    jl.seval(f'Random.seed!({seed})')
    jl.seval(f'test_data = randn(Float64, {dim}, {n})')
    jl.seval(f'test_index = ManifoldANN.build_index(ManifoldANN.BruteForceIndex, test_data)')

    # Build both directed and undirected graphs
    jl.seval(f'test_graph_directed = ManifoldANN.build_knn_graph(test_index, test_data; k={k}, directed=true)')
    jl.seval(f'test_graph_undirected = ManifoldANN.build_knn_graph(test_index, test_data; k={k}, directed=false)')

    # Estimate data scale for Sinkhorn regularization
    jl.seval('''
        sample_size = min(100, size(test_data, 2))
        distances = Float64[]
        for i in 1:sample_size
            for j in i+1:sample_size
                push!(distances, norm(test_data[:, i] - test_data[:, j]))
            end
        end
        data_scale = mean(distances)
    ''')
    data_scale = float(jl.data_scale)

    return data_scale


def extract_edge_curvatures(n, k, solver_code, method_config):
    """Extract edge curvatures for a specific configuration"""

    config_name, directed, cost_metric, denom_metric, exclude_endpoints = method_config

    # Select graph based on directedness required
    graph_var = 'test_graph_directed' if directed else 'test_graph_undirected'

    # Compute curvatures
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

    # Extract edge curvatures
    jl.seval('''
        edge_records = [(edge.src, edge.dst, result.curvature)
                       for (edge, result) in results]
    ''')
    edge_records = jl.edge_records

    edge_data = []
    for src, dst, curv in edge_records:
        edge_data.append({
            'source_node': int(src),
            'target_node': int(dst),
            'curvature': float(curv)
        })

    # Summary statistics
    curvatures = [e['curvature'] for e in edge_data]
    n_edges = len(curvatures)
    n_edges_pruned = sum(1 for c in curvatures if c < 0)
    mean_curvature = np.mean(curvatures) if curvatures else 0.0
    std_curvature = np.std(curvatures) if curvatures else 0.0

    print(f"    {n} nodes, {n_edges:,} edges | "
          f"mean={mean_curvature:.4f}, std={std_curvature:.4f}, "
          f"negative={n_edges_pruned} ({100*n_edges_pruned/n_edges:.1f}%) | "
          f"{elapsed:.3f}s")

    return edge_data, elapsed


def main():
    parser = argparse.ArgumentParser(description='Extract Edge-Level Curvatures')
    parser.add_argument('--sizes', type=str, default='1000',
                       help='Comma-separated graph sizes (default: 1000)')
    parser.add_argument('--k', type=str, default='20',
                       help='Comma-separated k values (number of neighbors, default: 20)')
    parser.add_argument('--dim', type=int, default=50,
                       help='Data dimensionality (default: 50)')
    parser.add_argument('--runs', type=int, default=3,
                       help='Number of runs per configuration (default: 3)')
    parser.add_argument('--solver', type=str, default='hungarian',
                       choices=['hungarian', 'networksimplex', 'lpreference', 'sinkhorn'],
                       help='OT solver to use (default: hungarian)')
    parser.add_argument('--method', type=str, default='standard',
                       choices=['standard', 'orcml'],
                       help='ORC method configuration (default: standard)')
    parser.add_argument('--output-dir', type=str, default='results/orc',
                       help='Output directory (default: results/orc)')

    args = parser.parse_args()

    # Parse sizes and k values
    sizes = [int(s.strip()) for s in args.sizes.split(',')]
    k_values = [int(k.strip()) for k in args.k.split(',')]

    # Create output directory
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Solver name mapping (code will be configured per-dataset for Sinkhorn)
    solver_names = {
        'hungarian': 'Hungarian',
        'networksimplex': 'NetworkSimplex',
        'lpreference': 'LPReference',
        'sinkhorn': 'Sinkhorn',
    }
    solver_name = solver_names[args.solver]

    # Method configuration
    method_configs = {
        'standard': ('ORC', True, 'euclidean', 'euclidean', False),
        'orcml': ('ORC-orcml', False, 'geodesic_normalized', 'normalized', True),
    }
    method_config = method_configs[args.method]
    method_name = method_config[0]

    print("="*80)
    print("EDGE CURVATURE EXTRACTION")
    print("="*80)
    print(f"Graph sizes: {sizes}")
    print(f"k values (neighbors): {k_values}")
    print(f"Dimensionality: {args.dim}")
    print(f"Runs per size: {args.runs}")
    print(f"Solver: {solver_name}")
    print(f"Method: {method_name}")
    print(f"Output directory: {output_dir}")
    print("="*80)

    # Setup Julia
    setup_julia()

    # Collect all edge data
    all_edge_data = []

    for n in sizes:
        for k in k_values:
            print(f"\nExtracting edges for n={n}, k={k}:")

            for run in range(args.runs):
                # Generate test data and get scale for Sinkhorn
                seed = 42 + run
                data_scale = generate_test_data(n, args.dim, k, seed)

                # Configure solver (Sinkhorn needs data-dependent regularization)
                if args.solver == 'sinkhorn':
                    sinkhorn_reg = 0.15 * data_scale
                    solver_code = f'ManifoldANN.SinkhornSolver(reg={sinkhorn_reg:.6f}, maxiter=2000, atol=1e-6)'
                elif args.solver == 'hungarian':
                    solver_code = 'ManifoldANN.HungarianSolver()'
                elif args.solver == 'networksimplex':
                    solver_code = 'ManifoldANN.NetworkSimplexSolver()'
                elif args.solver == 'lpreference':
                    solver_code = 'ManifoldANN.LPReferenceSolver()'

                # Extract edge curvatures
                print(f"  Run {run+1}/{args.runs}: ", end='', flush=True)
                edge_data, elapsed = extract_edge_curvatures(n, k, solver_code, method_config)

                # Add metadata to each edge
                for edge in edge_data:
                    edge.update({
                        'n': n,
                        'k': k,
                        'dim': args.dim,
                        'run': run,
                        'method': method_name,
                        'solver': solver_name
                    })

                all_edge_data.extend(edge_data)

    # Save edge curvatures
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    edges_file = output_dir / f"edge_curvatures_{args.method}_{args.solver}_{timestamp}.csv"

    with open(edges_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'n', 'k', 'dim', 'run', 'method', 'solver',
            'source_node', 'target_node', 'curvature'
        ])
        writer.writeheader()
        writer.writerows(all_edge_data)

    print(f"\n{'='*80}")
    print(f"Edge curvatures saved to: {edges_file}")
    print(f"Total edges extracted: {len(all_edge_data):,}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
