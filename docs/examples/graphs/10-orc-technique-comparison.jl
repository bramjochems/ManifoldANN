"""
# Example: Comparing Different ORC Computation Techniques

This example demonstrates and compares different approaches to computing
Ollivier-Ricci curvature:

1. **Original**: Directed graph, Euclidean distances
2. **orcml**: Undirected graph, geodesic distances with effective_epsilon
3. **Hybrid**: Geodesic costs with Euclidean denominator

Each technique has different characteristics and use cases.
"""

using ManifoldANN
using LinearAlgebra
using Random
using Statistics
using Printf

println("="^80)
println("Comparing ORC Computation Techniques")
println("="^80)

# Generate data
Random.seed!(42)
n = 500
d = 10

data = randn(d, n)

println("\nDataset: Random Gaussian")
println("  Points: $n")
println("  Dimension: $d")

# Build indices for different techniques
k = 15
brute = build_index(BruteForceIndex, data)

# Technique 1: Original (Directed, Euclidean)
println("\n" * "="^80)
println("Technique 1: Original ManifoldANN")
println("="^80)

println("\nConfiguration:")
println("  • Graph: Directed (standard k-NN)")
println("  • Cost matrix: Euclidean distances")
println("  • Denominator: Euclidean distance")
println("  • Edge endpoints: Included in neighborhoods")

graph_directed = build_knn_graph(brute, data; k=k, directed=true)

println("\nBuilding graph...")
println("  Nodes: $(length(graph_directed))")
println("  Edges: $(sum(length(neighbors) for neighbors in graph_directed))")
println("  Avg degree: $(mean([length(neighbors) for neighbors in graph_directed]))")

println("\nComputing curvatures...")
start = time()
curv_original = compute_all_curvatures(
    graph_directed, data;
    exclude_edge_endpoints=false,
    cost_metric=:euclidean,
    denominator_metric=:euclidean,
    solver=HungarianSolver(),
    fallback_solver=NetworkSimplexSolver(),
    use_threading=true
)
time_original = time() - start

curv_vals_original = [r.curvature for r in values(curv_original)]

@printf("\nResults:")
@printf("\n  Time: %.3f seconds", time_original)
@printf("\n  Edges computed: %d", length(curv_original))
@printf("\n  Mean κ: %.4f", mean(curv_vals_original))
@printf("\n  Std κ: %.4f", std(curv_vals_original))
@printf("\n  Min κ: %.4f", minimum(curv_vals_original))
@printf("\n  Max κ: %.4f\n", maximum(curv_vals_original))

# Technique 2: orcml (Undirected, Geodesic + effective_epsilon)
println("\n" * "="^80)
println("Technique 2: orcml (ICLR 2025)")
println("="^80)

println("\nConfiguration:")
println("  • Graph: Undirected (symmetrized)")
println("  • Cost matrix: Geodesic with effective_epsilon weights")
println("  • Denominator: effective_epsilon")
println("  • Edge endpoints: Excluded from neighborhoods")

graph_undirected = build_knn_graph(brute, data; k=k, directed=false)

println("\nBuilding graph...")
println("  Nodes: $(length(graph_undirected))")
println("  Edges (directed form): $(sum(length(neighbors) for neighbors in graph_undirected))")
println("  Avg degree: $(mean([length(neighbors) for neighbors in graph_undirected]))")
println("  Note: Undirected graph has higher avg degree")

println("\nComputing curvatures...")
start = time()
curv_orcml = compute_all_curvatures(
    graph_undirected, data;
    exclude_edge_endpoints=true,
    cost_metric=:geodesic_normalized,
    denominator_metric=:normalized,
    solver=HungarianSolver(),
    fallback_solver=NetworkSimplexSolver(),
    use_threading=true
)
time_orcml = time() - start

curv_vals_orcml = [r.curvature for r in values(curv_orcml)]

@printf("\nResults:")
@printf("\n  Time: %.3f seconds (%.1fx vs original)", time_orcml, time_orcml / time_original)
@printf("\n  Edges computed: %d", length(curv_orcml))
@printf("\n  Mean κ: %.4f", mean(curv_vals_orcml))
@printf("\n  Std κ: %.4f", std(curv_vals_orcml))
@printf("\n  Min κ: %.4f", minimum(curv_vals_orcml))
@printf("\n  Max κ: %.4f\n", maximum(curv_vals_orcml))

# Technique 3: Hybrid (Geodesic costs, Euclidean denominator)
println("\n" * "="^80)
println("Technique 3: Hybrid Geodesic")
println("="^80)

println("\nConfiguration:")
println("  • Graph: Directed")
println("  • Cost matrix: Geodesic with Euclidean weights")
println("  • Denominator: Euclidean distance")
println("  • Edge endpoints: Included")

println("\nComputing curvatures...")
start = time()
curv_hybrid = compute_all_curvatures(
    graph_directed, data;
    exclude_edge_endpoints=false,
    cost_metric=:geodesic_euclidean,
    denominator_metric=:euclidean,
    solver=HungarianSolver(),
    fallback_solver=NetworkSimplexSolver(),
    use_threading=true
)
time_hybrid = time() - start

curv_vals_hybrid = [r.curvature for r in values(curv_hybrid)]

@printf("\nResults:")
@printf("\n  Time: %.3f seconds (%.1fx vs original)", time_hybrid, time_hybrid / time_original)
@printf("\n  Edges computed: %d", length(curv_hybrid))
@printf("\n  Mean κ: %.4f", mean(curv_vals_hybrid))
@printf("\n  Std κ: %.4f", std(curv_vals_hybrid))
@printf("\n  Min κ: %.4f", minimum(curv_vals_hybrid))
@printf("\n  Max κ: %.4f\n", maximum(curv_vals_hybrid))

# Comparative Analysis
println("\n" * "="^80)
println("Comparative Analysis")
println("="^80)

println("\n1. Performance:")
println("  " * "-"^60)
@printf("  %-20s | %10s | %10s | %10s\n", "Technique", "Time (s)", "Edges", "ms/edge")
println("  " * "-"^60)
@printf("  %-20s | %10.3f | %10d | %10.3f\n",
        "Original", time_original, length(curv_original),
        time_original / length(curv_original) * 1000)
@printf("  %-20s | %10.3f | %10d | %10.3f\n",
        "orcml", time_orcml, length(curv_orcml),
        time_orcml / length(curv_orcml) * 1000)
@printf("  %-20s | %10.3f | %10d | %10.3f\n",
        "Hybrid", time_hybrid, length(curv_hybrid),
        time_hybrid / length(curv_hybrid) * 1000)

println("\n2. Curvature Statistics:")
println("  " * "-"^70)
@printf("  %-20s | %10s | %10s | %10s | %10s\n",
        "Technique", "Mean κ", "Std κ", "Min κ", "Max κ")
println("  " * "-"^70)
@printf("  %-20s | %10.4f | %10.4f | %10.4f | %10.4f\n",
        "Original", mean(curv_vals_original), std(curv_vals_original),
        minimum(curv_vals_original), maximum(curv_vals_original))
@printf("  %-20s | %10.4f | %10.4f | %10.4f | %10.4f\n",
        "orcml", mean(curv_vals_orcml), std(curv_vals_orcml),
        minimum(curv_vals_orcml), maximum(curv_vals_orcml))
@printf("  %-20s | %10.4f | %10.4f | %10.4f | %10.4f\n",
        "Hybrid", mean(curv_vals_hybrid), std(curv_vals_hybrid),
        minimum(curv_vals_hybrid), maximum(curv_vals_hybrid))

println("\n3. Negative Curvature (Manifold Cuts):")
println("  " * "-"^60)
@printf("  %-20s | %10s | %10s\n", "Technique", "Count", "Percentage")
println("  " * "-"^60)

for (name, vals) in [("Original", curv_vals_original),
                      ("orcml", curv_vals_orcml),
                      ("Hybrid", curv_vals_hybrid)]
    neg_count = sum(vals .< 0)
    @printf("  %-20s | %10d | %9.1f%%\n",
            name, neg_count, 100 * neg_count / length(vals))
end

# Recommendations
println("\n" * "="^80)
println("Recommendations by Use Case")
println("="^80)

println("""
1. **Fast Graph Filtering** (Original)
   Use when: Speed is critical, approximate filtering is acceptable
   Configuration:
     directed=true
     cost_metric=:euclidean
     denominator_metric=:euclidean
     exclude_edge_endpoints=false
   Performance: Fastest (baseline)
   Accuracy: Good for general graph refinement

2. **Research Replication** (orcml)
   Use when: Need to match published results (ICLR 2025)
   Configuration:
     directed=false
     cost_metric=:geodesic_normalized
     denominator_metric=:normalized
     exclude_edge_endpoints=true
   Performance: $(round(time_orcml / time_original, digits=1))x slower
   Accuracy: 99.65% correlation with orcml.py
   Benefits: Scale-invariant, theoretically justified

3. **Hybrid Geodesic** (Experimental)
   Use when: Want geodesic distances without full orcml overhead
   Configuration:
     directed=true (or false)
     cost_metric=:geodesic_euclidean
     denominator_metric=:euclidean
     exclude_edge_endpoints=false
   Performance: $(round(time_hybrid / time_original, digits=1))x slower
   Benefits: Better than Euclidean, faster than full orcml

Solver Recommendations:
  • Primary: HungarianSolver (fast for uniform distributions)
  • Fallback: NetworkSimplexSolver (robust, no NaN issues)
  • Avoid: SinkhornSolver (100% failure rate with default parameters)
""")

println("="^80)
