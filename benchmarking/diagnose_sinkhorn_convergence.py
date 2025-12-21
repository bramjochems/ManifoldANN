#!/usr/bin/env python3
"""
Diagnose Sinkhorn Convergence Issues

Tests different (n, k) combinations with various regularization parameters
to understand convergence behavior before running full benchmarks.

Usage:
    python diagnose_sinkhorn_convergence.py
"""

import os
import sys
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

    # Activate the root ManifoldANN project
    root_dir = Path(__file__).parent.parent
    jl.seval(f'Pkg.activate("{root_dir}")')

    jl.seval('using ManifoldANN')
    jl.seval('using LinearAlgebra')
    jl.seval('using Random')
    jl.seval('using Statistics')
    print("✓ Julia initialized\n")


def generate_test_data(n, dim, k, seed=42):
    """Generate random test data and k-NN graph"""
    jl.seval(f'Random.seed!({seed})')
    jl.seval(f'test_data = randn(Float64, {dim}, {n})')
    jl.seval(f'test_index = ManifoldANN.build_index(ManifoldANN.BruteForceIndex, test_data)')
    jl.seval(f'test_graph = ManifoldANN.build_knn_graph(test_index, test_data; k={k}, directed=true)')

    # Compute data scale
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


def test_sinkhorn_convergence(n, k, reg_value, dim=50):
    """
    Test Sinkhorn convergence for a given configuration.

    Returns:
        dict with convergence statistics
    """
    # Generate data
    data_scale = generate_test_data(n, dim, k, seed=42)

    # Run Sinkhorn with specified regularization
    jl.seval(f'''
        solver = ManifoldANN.SinkhornSolver(reg={reg_value}, maxiter=2000, atol=1e-6)

        # Capture warnings by redirecting stderr
        original_stderr = stderr
        (rd, wr) = redirect_stderr()

        results = ManifoldANN.compute_all_curvatures(
            test_graph, test_data;
            cost_metric=:euclidean,
            denominator_metric=:euclidean,
            exclude_edge_endpoints=false,
            solver=solver,
            fallback_solver=ManifoldANN.NetworkSimplexSolver()
        )

        # Restore stderr and capture warnings
        redirect_stderr(original_stderr)
        close(wr)
        warning_output = String(read(rd))
        close(rd)

        # Count edges and check for convergence warnings
        n_edges = length(results)
        has_warnings = occursin("did not converge", warning_output)

        # Count how many edges used fallback (NetworkSimplex)
        n_fallback = count(r -> r.solver_type == :network_simplex, values(results))
        n_sinkhorn = count(r -> r.solver_type == :sinkhorn, values(results))
    ''')

    n_edges = int(jl.n_edges)
    has_warnings = bool(jl.has_warnings)
    n_fallback = int(jl.n_fallback)
    n_sinkhorn = int(jl.n_sinkhorn)

    return {
        'n': n,
        'k': k,
        'data_scale': data_scale,
        'reg_value': reg_value,
        'reg_ratio': reg_value / data_scale,
        'n_edges': n_edges,
        'n_sinkhorn': n_sinkhorn,
        'n_fallback': n_fallback,
        'has_warnings': has_warnings,
        'convergence_rate': n_sinkhorn / n_edges if n_edges > 0 else 0.0
    }


def main():
    # Test configurations
    sizes = [200, 400, 800, 1600]
    k_values = [5, 10, 20]

    # Regularization multipliers to test (as percentage of data_scale)
    reg_multipliers = [0.05, 0.10, 0.15, 0.20, 0.25]

    print("="*80)
    print("SINKHORN CONVERGENCE DIAGNOSTIC")
    print("="*80)
    print(f"Testing sizes: {sizes}")
    print(f"Testing k values: {k_values}")
    print(f"Regularization multipliers: {reg_multipliers}")
    print(f"(reg = multiplier × data_scale)")
    print("="*80)
    print()

    # Setup Julia
    setup_julia()

    # Results storage
    all_results = []

    for n in sizes:
        for k in k_values:
            print(f"\n{'='*80}")
            print(f"Testing n={n}, k={k}")
            print(f"{'='*80}")

            # First, estimate data scale
            data_scale = generate_test_data(n, 50, k, seed=42)
            print(f"Data scale: {data_scale:.4f}")
            print()

            print(f"{'Multiplier':<12} {'Reg Value':<12} {'Converged':<12} {'Fallback':<12} {'Conv Rate':<12}")
            print("-"*80)

            for mult in reg_multipliers:
                reg_value = mult * data_scale

                try:
                    result = test_sinkhorn_convergence(n, k, reg_value)
                    all_results.append(result)

                    conv_str = f"{result['n_sinkhorn']}/{result['n_edges']}"
                    fall_str = f"{result['n_fallback']}/{result['n_edges']}"
                    rate_str = f"{result['convergence_rate']*100:.1f}%"
                    warn_marker = "⚠️ " if result['has_warnings'] else "✓ "

                    print(f"{warn_marker}{mult:<10.2f} {reg_value:<12.4f} {conv_str:<12} {fall_str:<12} {rate_str:<12}")

                except Exception as e:
                    print(f"✗ {mult:<10.2f} ERROR: {e}")

    # Summary analysis
    print("\n" + "="*80)
    print("SUMMARY ANALYSIS")
    print("="*80)

    # Group by (n, k) and find best regularization
    from collections import defaultdict
    grouped = defaultdict(list)

    for result in all_results:
        key = (result['n'], result['k'])
        grouped[key].append(result)

    print(f"\n{'n':<8} {'k':<8} {'Best Mult':<12} {'Best Reg':<12} {'Conv Rate':<12}")
    print("-"*80)

    for (n, k), results in sorted(grouped.items()):
        # Find best convergence rate
        best = max(results, key=lambda r: r['convergence_rate'])

        if best['convergence_rate'] == 1.0:
            marker = "✓"
        elif best['convergence_rate'] >= 0.95:
            marker = "~"
        else:
            marker = "✗"

        print(f"{marker} {n:<6} {k:<6} {best['reg_ratio']:<12.2f} {best['reg_value']:<12.4f} {best['convergence_rate']*100:<11.1f}%")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Find multiplier that works best across all configurations
    from collections import Counter
    best_by_config = {}
    for (n, k), results in grouped.items():
        best = max(results, key=lambda r: r['convergence_rate'])
        best_by_config[(n, k)] = best['reg_ratio']

    # Count which multiplier works best most often
    multiplier_votes = Counter(best_by_config.values())
    recommended_mult = max(multiplier_votes, key=multiplier_votes.get)

    print(f"\nRecommended regularization: {recommended_mult:.2f} × data_scale")
    print(f"This works best for {multiplier_votes[recommended_mult]}/{len(best_by_config)} configurations")

    # Show which configs need fallback
    problematic = [(n, k) for (n, k), mult in best_by_config.items()
                   if grouped[(n, k)][0]['convergence_rate'] < 1.0]

    if problematic:
        print(f"\nConfigurations with convergence issues: {problematic}")
        print("These will use fallback solver (NetworkSimplex) for some edges.")
    else:
        print("\n✓ All configurations achieve 100% convergence with proper regularization!")


if __name__ == '__main__':
    main()
