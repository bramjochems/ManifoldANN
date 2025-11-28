#=
Example: Curvature-Based Graph Filtering

This example demonstrates how to compute Ollivier-Ricci curvature on kNN graph
edges and filter out edges that likely cut across the manifold.

Higher curvature edges follow the manifold structure better, while low curvature
edges may represent shortcuts across the manifold.
=#

using ManifoldANN
using LinearAlgebra
using Random

println("=== Curvature-Based Graph Filtering Example ===\n")

# Generate synthetic manifold data: points on a 2D grid (manifold) embedded in 10D
Random.seed!(42)
n_grid = 15  # 15x15 grid = 225 points
grid_points = [[Float64(i), Float64(j)] for i in 1:n_grid for j in 1:n_grid]
n_points = length(grid_points)

# Embed in 10D with a smooth nonlinear mapping + small noise
data_10d = zeros(10, n_points)
for (idx, point) in enumerate(grid_points)
    x, y = point
    # Nonlinear embedding (e.g., Swiss roll-like)
    data_10d[1, idx] = x
    data_10d[2, idx] = y
    data_10d[3, idx] = sin(x / 3)
    data_10d[4, idx] = cos(y / 3)
    data_10d[5, idx] = x * y / 50
    # Remaining dimensions: small noise
    data_10d[6:10, idx] .= randn(5) .* 0.1
end

println("Generated $(n_points) points on a 2D grid embedded in 10D")
println("Data shape: $(size(data_10d))\n")

# ============================================================================
# Build initial kNN graph
# ============================================================================

println("--- Building Initial kNN Graph ---")
k = 8
index = build_index(BruteForceIndex, data_10d)
graph = build_knn_graph(index, data_10d; k=k)

total_edges = sum(length(graph[i]) for i in 1:n_points)
println("k = $(graph.k)")
println("Total edges: $(total_edges)")
println()

# ============================================================================
# Compute curvatures for all edges
# ============================================================================

println("--- Computing Edge Curvatures ---")

curvatures = compute_all_curvatures(graph, data_10d)
println("Computed curvature for $(length(curvatures)) edges")

# Analyze curvature distribution
stats = curvature_statistics(curvatures)
println("\nCurvature Statistics:")
println("  Mean:     $(round(stats.mean, digits=3))")
println("  Std:      $(round(stats.std, digits=3))")
println("  Min:      $(round(stats.min, digits=3))")
println("  Max:      $(round(stats.max, digits=3))")
println("  Median:   $(round(stats.median, digits=3))")
println("  Positive: $(stats.n_positive) / $(stats.n_positive + stats.n_negative)")
println("  Negative: $(stats.n_negative) / $(stats.n_positive + stats.n_negative)")
println()

# ============================================================================
# Filter graph by curvature threshold
# ============================================================================

println("--- Filtering Graph (threshold = 0.0) ---")

filtered_graph_0 = filter_graph(
    graph, data_10d,
    curvature_threshold=0.0,
    min_neighbors=3,
    verbose=false
)

filtered_edges_0 = sum(length(filtered_graph_0[i]) for i in 1:n_points)
println("Original edges: $(total_edges)")
println("Filtered edges: $(filtered_edges_0)")
println("Removed: $(total_edges - filtered_edges_0) edges ($(round(100 * (total_edges - filtered_edges_0) / total_edges, digits=1))%)")
println()

# ============================================================================
# More aggressive filtering
# ============================================================================

println("--- Aggressive Filtering (threshold = 0.3) ---")

filtered_graph_03 = filter_graph(
    graph, data_10d,
    curvature_threshold=0.3,
    min_neighbors=3,
    verbose=false
)

filtered_edges_03 = sum(length(filtered_graph_03[i]) for i in 1:n_points)
println("Original edges: $(total_edges)")
println("Filtered edges: $(filtered_edges_03)")
println("Removed: $(total_edges - filtered_edges_03) edges ($(round(100 * (total_edges - filtered_edges_03) / total_edges, digits=1))%)")
println()

# ============================================================================
# Curvature solver comparison
# ============================================================================

println("--- Solver Comparison ---")

# Sample a few edges and compute with different solvers
sample_edges = [
    (1, graph[1][1]),
    (50, graph[50][1]),
    (100, graph[100][1])
]

distance_fn = (i, j) -> norm(data_10d[:, i] - data_10d[:, j])

for (x, y) in sample_edges
    x_nb = uniform_neighborhood(x, graph[x], Float64)
    y_nb = uniform_neighborhood(y, graph[y], Float64)
    edge_dist = distance_fn(x, y)
    edge_view = create_edge_view(x_nb, y_nb, edge_dist)

    # FastMatchingSolver
    if can_handle(FastMatchingSolver(), edge_view)
        result_fast = compute_curvature(FastMatchingSolver(), edge_view, distance_fn)
        println("Edge ($x, $y) [FastMatching]: κ = $(round(result_fast.curvature, digits=3))")
    else
        println("Edge ($x, $y): FastMatching cannot handle (degrees differ or non-uniform)")
    end

    # GenericOTSolver
    result_generic = compute_curvature(GenericOTSolver(), edge_view, distance_fn)
    println("Edge ($x, $y) [GenericOT]:     κ = $(round(result_generic.curvature, digits=3))")
    println()
end

# ============================================================================
# Effect on graph properties
# ============================================================================

println("--- Graph Property Comparison ---")

function graph_summary(g, name)
    degrees = [length(g[i]) for i in 1:length(g)]
    println("$(name):")
    println("  k (nominal):  $(g.k)")
    println("  Mean degree:  $(round(sum(degrees) / length(degrees), digits=2))")
    println("  Min degree:   $(minimum(degrees))")
    println("  Max degree:   $(maximum(degrees))")
end

graph_summary(graph, "Original")
graph_summary(filtered_graph_0, "Filtered (κ ≥ 0.0)")
graph_summary(filtered_graph_03, "Filtered (κ ≥ 0.3)")
println()

println("=== Example Complete ===")
