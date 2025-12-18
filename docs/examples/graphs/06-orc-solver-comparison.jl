#!/usr/bin/env julia
"""
# Ollivier-Ricci Curvature (ORC) Solver Comparison

This example demonstrates the different ORC solvers available in ManifoldANN
and helps you choose the right solver for your use case.

## Available Solvers

1. **HungarianSolver** - Fast exact solver for uniform distributions
2. **SinkhornSolver** - Fast approximate solver for any distribution
3. **NetworkSimplexSolver** - Exact solver for any distribution (slower)
4. **GreedySolver** - Fastest approximate solver
5. **LPReferenceSolver** - Generic LP solver (reference implementation)

## ORC Formula

For an edge (x,y), the Ollivier-Ricci curvature is:

    κ(x,y) = 1 - W₁(μₓ, μᵧ) / d(x,y)

where:
- W₁(μₓ, μᵧ) is the Wasserstein distance between neighborhood distributions
- d(x,y) is the edge distance (NOT shortest path!)
- μₓ, μᵧ are probability distributions over neighborhoods

## When to Use Each Solver

- **Hungarian**: Best for k-NN graphs with uniform neighborhoods (default)
- **Sinkhorn**: Fast approximate for any distribution, good accuracy
- **NetworkSimplex**: Exact for non-uniform distributions, slower
- **Greedy**: When speed matters more than accuracy
- **LP Reference**: Debugging and verification only
"""

using ManifoldANN
using LinearAlgebra
using Random
using Statistics
using Printf

println("="^70)
println("ORC Solver Comparison Example")
println("="^70)

# ============================================================================
# 1. Generate Test Data
# ============================================================================

println("\n## 1. Generate Test Data")

Random.seed!(42)
n_nodes = 200
k = 8
dim = 10

data = randn(Float64, dim, n_nodes)
idx = build_index(BruteForceIndex, data)
graph = build_knn_graph(idx, data; k=k)

println("  Nodes: $n_nodes")
println("  k-NN: $k")
println("  Dimensions: $dim")
println("  Total edges: $(sum(length(graph[i]) for i in 1:n_nodes))")

# ============================================================================
# 2. Solver Comparison
# ============================================================================

println("\n## 2. Solver Performance Comparison")
println()

solvers = [
    ("Hungarian", HungarianSolver(), "Fast exact (uniform neighborhoods)"),
    ("Sinkhorn", SinkhornSolver(reg=0.01), "Fast approximate (any distribution)"),
    ("NetworkSimplex", NetworkSimplexSolver(), "Slow exact (any distribution)"),
    ("Greedy", GreedySolver(), "Fastest approximate"),
]

results_dict = Dict{String, Dict{Tuple{Int,Int}, CurvatureResult{Float64}}}()

for (name, solver, description) in solvers
    println("### $name")
    println("    $description")

    # Warmup
    _ = compute_all_curvatures(graph, data; solver=solver, fallback_solver=solver)

    # Benchmark
    times = Float64[]
    for _ in 1:3
        t = @elapsed begin
            results_dict[name] = compute_all_curvatures(
                graph, data,
                solver=solver,
                fallback_solver=solver
            )
        end
        push!(times, t)
    end

    median_time = median(times)
    n_edges = length(results_dict[name])
    per_edge = median_time * 1e6 / n_edges

    @printf("    Time: %.2f ms (%.2f μs/edge)\n", median_time * 1000, per_edge)
    @printf("    Edges computed: %d\n\n", n_edges)
end

# ============================================================================
# 3. Accuracy Comparison
# ============================================================================

println("## 3. Accuracy Comparison")
println()

# Compare against Hungarian (exact for uniform neighborhoods)
reference = results_dict["Hungarian"]

for (name, _, _) in solvers
    if name == "Hungarian"
        continue
    end

    diffs = Float64[]
    for edge in keys(reference)
        if haskey(results_dict[name], edge)
            κ_ref = reference[edge].curvature
            κ_test = results_dict[name][edge].curvature
            push!(diffs, abs(κ_ref - κ_test))
        end
    end

    if !isempty(diffs)
        @printf("  %s vs Hungarian:\n", name)
        @printf("    Mean difference: %.2e\n", mean(diffs))
        @printf("    Max difference:  %.2e\n", maximum(diffs))
        @printf("    Median difference: %.2e\n\n", median(diffs))
    end
end

# ============================================================================
# 4. Example: Computing Curvature for a Single Edge
# ============================================================================

println("## 4. Example: Single Edge Computation")
println()

# Pick a random edge
x = 1
y = graph[1][1]
edge_dist = norm(data[:, x] - data[:, y])

println("  Computing curvature for edge ($x, $y)")
println("  Edge distance: $edge_dist")

# Build neighborhoods
neighborhoods = Dict{Int,NodeNeighborhood{Float64}}(
    i => uniform_neighborhood(i, graph[i], Float64) for i in 1:n_nodes
)

edge_view = create_edge_view(neighborhoods[x], neighborhoods[y], edge_dist)

println("  N($x) size: $(length(edge_view.shared) + length(edge_view.unique_x))")
println("  N($y) size: $(length(edge_view.shared) + length(edge_view.unique_y))")
println("  Shared nodes: $(length(edge_view.shared))")
println()

# Compute with each solver
for (name, solver, _) in solvers
    if can_handle(solver, edge_view)
        result = compute_curvature(solver, edge_view, (i, j) -> norm(data[:, i] - data[:, j]))
        @printf("  %s:\n", name)
        @printf("    κ = %.6f\n", result.curvature)
        @printf("    W₁ = %.6f\n", result.wasserstein_distance)
        @printf("    d = %.6f\n\n", result.edge_distance)
    else
        println("  $name: Cannot handle this edge (requirements not met)\n")
    end
end

# ============================================================================
# 5. Recommendations
# ============================================================================

println("## 5. Recommendations")
println()
println("### For k-NN Graphs (uniform neighborhoods):")
println("  ✓ Use HungarianSolver() - fast and exact")
println()
println("### For Weighted/Non-uniform Graphs:")
println("  ✓ Use SinkhornSolver() for speed")
println("  ✓ Use NetworkSimplexSolver() for accuracy")
println()
println("### For Maximum Speed (approximate OK):")
println("  ✓ Use GreedySolver()")
println()
println("### Default Configuration:")
println()
println("```julia")
println("curvatures = compute_all_curvatures(")
println("    graph, data,")
println("    solver=HungarianSolver(),      # Try exact first")
println("    fallback_solver=SinkhornSolver()  # Fall back if needed")
println(")")
println("```")
println()

# ============================================================================
# 6. Curvature Statistics
# ============================================================================

println("## 6. Curvature Statistics")
println()

stats = curvature_statistics(results_dict["Hungarian"])

@printf("  Mean curvature: %.4f\n", stats.mean)
@printf("  Std deviation:  %.4f\n", stats.std)
@printf("  Min curvature:  %.4f\n", stats.min)
@printf("  Max curvature:  %.4f\n", stats.max)
@printf("  Median:         %.4f\n", stats.median)
@printf("  Positive edges: %d (%.1f%%)\n", stats.n_positive,
        100 * stats.n_positive / (stats.n_positive + stats.n_negative))
@printf("  Negative edges: %d (%.1f%%)\n", stats.n_negative,
        100 * stats.n_negative / (stats.n_positive + stats.n_negative))

println("\n✓ Example complete!")
