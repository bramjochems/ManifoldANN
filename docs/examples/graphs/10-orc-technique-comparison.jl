"""
# Example: Comparing ORC Computation Variants

This example demonstrates and compares the two supported variants for
computing Ollivier-Ricci curvature:

1. **Standard ORC** (`StandardORC()`): Directed graph, Euclidean
   distances throughout, neighbourhoods include the edge endpoints.
2. **ORC-ManL** (`ORCManL()`): Undirected graph, geodesic-normalised
   cost, effective-epsilon denominator, neighbourhoods exclude the
   edge endpoints (the variant from the orcml paper, ICLR 2025).

Each variant has different characteristics and use cases.
"""

using ManifoldANN
using LinearAlgebra
using Random
using Statistics
using Printf

println("="^80)
println("Comparing ORC Variants")
println("="^80)

# Generate data
Random.seed!(42)
n = 500
d = 10

data = randn(d, n)

println("\nDataset: Random Gaussian")
println("  Points: $n")
println("  Dimension: $d")

# Build indices for both variants
k = 15
brute = build_index(BruteForceIndex, data)

# Variant 1: Standard ORC (Directed, Euclidean)
println("\n" * "="^80)
println("Variant 1: Standard ORC (StandardORC)")
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
    variant=StandardORC(),
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

# Variant 2: ORC-ManL (Undirected, Geodesic + effective_epsilon)
println("\n" * "="^80)
println("Variant 2: ORC-ManL (ORCManL, ICLR 2025)")
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
    variant=ORCManL(),
    solver=HungarianSolver(),
    fallback_solver=NetworkSimplexSolver(),
    use_threading=true
)
time_orcml = time() - start

curv_vals_orcml = [r.curvature for r in values(curv_orcml)]

@printf("\nResults:")
@printf("\n  Time: %.3f seconds (%.1fx vs standard)", time_orcml, time_orcml / time_original)
@printf("\n  Edges computed: %d", length(curv_orcml))
@printf("\n  Mean κ: %.4f", mean(curv_vals_orcml))
@printf("\n  Std κ: %.4f", std(curv_vals_orcml))
@printf("\n  Min κ: %.4f", minimum(curv_vals_orcml))
@printf("\n  Max κ: %.4f\n", maximum(curv_vals_orcml))

# Comparative Analysis
println("\n" * "="^80)
println("Comparative Analysis")
println("="^80)

println("\n1. Performance:")
println("  " * "-"^60)
@printf("  %-20s | %10s | %10s | %10s\n", "Variant", "Time (s)", "Edges", "ms/edge")
println("  " * "-"^60)
@printf("  %-20s | %10.3f | %10d | %10.3f\n",
        "StandardORC", time_original, length(curv_original),
        time_original / length(curv_original) * 1000)
@printf("  %-20s | %10.3f | %10d | %10.3f\n",
        "ORCManL", time_orcml, length(curv_orcml),
        time_orcml / length(curv_orcml) * 1000)

println("\n2. Curvature Statistics:")
println("  " * "-"^70)
@printf("  %-20s | %10s | %10s | %10s | %10s\n",
        "Variant", "Mean κ", "Std κ", "Min κ", "Max κ")
println("  " * "-"^70)
@printf("  %-20s | %10.4f | %10.4f | %10.4f | %10.4f\n",
        "StandardORC", mean(curv_vals_original), std(curv_vals_original),
        minimum(curv_vals_original), maximum(curv_vals_original))
@printf("  %-20s | %10.4f | %10.4f | %10.4f | %10.4f\n",
        "ORCManL", mean(curv_vals_orcml), std(curv_vals_orcml),
        minimum(curv_vals_orcml), maximum(curv_vals_orcml))

println("\n3. Negative Curvature (Manifold Cuts):")
println("  " * "-"^60)
@printf("  %-20s | %10s | %10s\n", "Variant", "Count", "Percentage")
println("  " * "-"^60)

for (name, vals) in [("StandardORC", curv_vals_original),
                      ("ORCManL", curv_vals_orcml)]
    neg_count = sum(vals .< 0)
    @printf("  %-20s | %10d | %9.1f%%\n",
            name, neg_count, 100 * neg_count / length(vals))
end

# Recommendations
println("\n" * "="^80)
println("Recommendations by Use Case")
println("="^80)

println("""
1. **Fast Graph Filtering** (StandardORC)
   Use when: Speed is critical, approximate filtering is acceptable
   Code:
     graph = build_knn_graph(index, data; k=k, directed=true)
     compute_all_curvatures(graph, data; variant=StandardORC())
   Performance: Fastest (baseline)
   Accuracy: Good for general graph refinement

2. **Research Replication** (ORCManL)
   Use when: Need to match published results (ICLR 2025)
   Code:
     graph = build_knn_graph(index, data; k=k, directed=false)
     compute_all_curvatures(graph, data; variant=ORCManL())
   For bit-for-bit match with the reference Python `orcml`:
     compute_all_curvatures(graph, data;
                            variant=ORCManL(profile=OrcmlExact()))
   Performance: $(round(time_orcml / time_original, digits=1))x slower
   Accuracy: ≥99.65% correlation with orcml.py
   Benefits: Scale-invariant, theoretically justified

Solver Recommendations:
  • Primary: HungarianSolver (fast for uniform distributions)
  • Fallback: NetworkSimplexSolver (robust, no NaN issues)
  • Avoid: SinkhornSolver (100% failure rate with default parameters)
""")

println("="^80)
