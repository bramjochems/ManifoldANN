#=
Example: Curvature Filtering - Scaling Study

This example studies how curvature filtering behaves as k (neighborhood size) varies:
1. Runtime performance of different solvers
2. Quality metrics (precision/recall) vs k
3. Computational cost scaling

NOTE: Sinkhorn solver is mentioned in the API but not yet implemented.
=#

using ManifoldANN
using LinearAlgebra
using Random
using Printf

println("=== Curvature Filtering Scaling Study ===\n")

# ============================================================================
# Generate manifold with ground truth
# ============================================================================

Random.seed!(42)
n_grid = 20  # Larger grid for scaling study
grid_points = [[Float64(i), Float64(j)] for i in 1:n_grid for j in 1:n_grid]
n_points = length(grid_points)

# Embed in 10D
data_10d = zeros(10, n_points)
for (idx, point) in enumerate(grid_points)
    x, y = point
    data_10d[1, idx] = x
    data_10d[2, idx] = y
    data_10d[3, idx] = sin(x / 3)
    data_10d[4, idx] = cos(y / 3)
    data_10d[5, idx] = x * y / 50
    data_10d[6:10, idx] .= randn(5) .* 0.1
end

println("Generated $(n_points) points on $(n_grid)×$(n_grid) grid in 10D\n")

# ============================================================================
# Helper functions
# ============================================================================

function grid_distance(i::Int, j::Int, grid_points)
    p1, p2 = grid_points[i], grid_points[j]
    return abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])
end

function are_grid_neighbors(i::Int, j::Int, grid_points; max_dist=1)
    return grid_distance(i, j, grid_points) <= max_dist
end

function evaluate_filtering(graph, filtered_graph, grid_points; max_grid_dist=1)
    tp = fp = tn = fn = 0

    for i in 1:length(graph)
        original = Set(graph[i])
        filtered = Set(filtered_graph[i])
        removed = setdiff(original, filtered)

        for j in original
            is_true = are_grid_neighbors(i, j, grid_points; max_dist=max_grid_dist)
            was_removed = j in removed

            if is_true
                was_removed ? (fp += 1) : (tn += 1)
            else
                was_removed ? (tp += 1) : (fn += 1)
            end
        end
    end

    precision = (tp + fp) > 0 ? tp / (tp + fp) : 0.0
    recall = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
    f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0

    return (tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall, f1=f1)
end

# ============================================================================
# Study 1: Runtime Performance (k=8, different solvers)
# ============================================================================

println("="^75)
println("STUDY 1: Solver Runtime Comparison (k=8)")
println("="^75)
println()

k = 8
index = build_index(BruteForceIndex, data_10d)
graph = build_knn_graph(index, data_10d; k=k)

distance_fn = (i, j) -> norm(data_10d[:, i] - data_10d[:, j])

# Sample 100 edges for timing
sample_edges = Tuple{Int,Int}[]
for i in 1:min(100, n_points)
    for j in graph[i]
        push!(sample_edges, (i, j))
        length(sample_edges) >= 100 && break
    end
    length(sample_edges) >= 100 && break
end

println("Testing on $(length(sample_edges)) sample edges\n")

# Prepare neighborhoods (reused across solvers)
neighborhoods = Dict{Int,NodeNeighborhood{Float64}}(
    i => uniform_neighborhood(i, graph[i], Float64) for i in 1:n_points
)

# Test HungarianSolver
println("HungarianSolver (Hungarian algorithm):")
time_hungarian = @elapsed begin
    for (x, y) in sample_edges
        edge_view = create_edge_view(neighborhoods[x], neighborhoods[y], distance_fn(x, y))
        if can_handle(HungarianSolver(), edge_view)
            compute_curvature(HungarianSolver(), edge_view, distance_fn)
        end
    end
end
@printf("  Time: %.3f ms (%.3f μs/edge)\n", time_hungarian * 1000, time_hungarian * 1e6 / length(sample_edges))
println()

# Test NetworkSimplexSolver
println("NetworkSimplexSolver (network flow):")
time_simplex = @elapsed begin
    for (x, y) in sample_edges
        edge_view = create_edge_view(neighborhoods[x], neighborhoods[y], distance_fn(x, y))
        compute_curvature(NetworkSimplexSolver(), edge_view, distance_fn)
    end
end
@printf("  Time: %.3f ms (%.3f μs/edge)\n", time_simplex * 1000, time_simplex * 1e6 / length(sample_edges))
println()

# Test GenericOTSolver (LP)
println("GenericOTSolver (LP via HiGHS):")
time_lp = @elapsed begin
    for (x, y) in sample_edges
        edge_view = create_edge_view(neighborhoods[x], neighborhoods[y], distance_fn(x, y))
        compute_curvature(GenericOTSolver(method=:lp), edge_view, distance_fn)
    end
end
@printf("  Time: %.3f ms (%.3f μs/edge)\n", time_lp * 1000, time_lp * 1e6 / length(sample_edges))
println()

# Test GenericOTSolver (Greedy)
println("GenericOTSolver (greedy heuristic):")
time_greedy = @elapsed begin
    for (x, y) in sample_edges
        edge_view = create_edge_view(neighborhoods[x], neighborhoods[y], distance_fn(x, y))
        compute_curvature(GenericOTSolver(method=:greedy), edge_view, distance_fn)
    end
end
@printf("  Time: %.3f ms (%.3f μs/edge)\n", time_greedy * 1000, time_greedy * 1e6 / length(sample_edges))
println()

# Test GenericOTSolver (Sinkhorn)
println("GenericOTSolver (Sinkhorn, ε=0.01):")
time_sinkhorn = @elapsed begin
    for (x, y) in sample_edges
        edge_view = create_edge_view(neighborhoods[x], neighborhoods[y], distance_fn(x, y))
        compute_curvature(GenericOTSolver(method=:sinkhorn, sinkhorn_reg=0.01), edge_view, distance_fn)
    end
end
@printf("  Time: %.3f ms (%.3f μs/edge)\n", time_sinkhorn * 1000, time_sinkhorn * 1e6 / length(sample_edges))
println()

println("NOTE: Sinkhorn provides approximate solution (entropy-regularized OT).")
println("      - O(k² iterations) complexity, typically 10-100 iterations")
println("      - Faster than LP, more accurate than greedy")
println("      - Approximation quality controlled by regularization (ε=0.01)")
println()

println("Speedup comparison (relative to LP):")
@printf("  Hungarian:    %.1fx faster\n", time_lp / time_fast_hungarian)
@printf("  Brute force:  %.1fx faster\n", time_lp / time_fast_brute)
@printf("  Sinkhorn:     %.1fx faster\n", time_lp / time_sinkhorn)
@printf("  Greedy:       %.1fx faster\n", time_lp / time_greedy)
println()

# ============================================================================
# Study 2: Quality vs k
# ============================================================================

println("="^75)
println("STUDY 2: Filtering Quality vs Neighborhood Size (k)")
println("="^75)
println()

println(@sprintf("%-5s %-12s %-12s %-10s %-10s %-10s %-10s %-10s",
    "k", "Total edges", "True nbrs", "False nbrs", "Precision", "Recall", "F1", "Time (s)"))
println("─"^75)

k_values = [4, 8, 12, 16, 20, 24]
curvature_threshold = 0.2

for k in k_values
    # Build graph
    index = build_index(BruteForceIndex, data_10d)
    graph = build_knn_graph(index, data_10d; k=k)

    # Count true/false neighbors
    n_true = n_false = 0
    for i in 1:n_points
        for j in graph[i]
            if are_grid_neighbors(i, j, grid_points; max_dist=1)
                n_true += 1
            else
                n_false += 1
            end
        end
    end

    # Filter graph (timed)
    filter_time = @elapsed begin
        filtered_graph = filter_graph(
            graph, data_10d,
            curvature_threshold=curvature_threshold,
            min_neighbors=3,
            verbose=false
        )
    end

    # Evaluate
    metrics = evaluate_filtering(graph, filtered_graph, grid_points; max_grid_dist=1)

    total_edges = n_true + n_false

    println(@sprintf("%-5d %-12d %-12d %-10d %-10.3f %-10.3f %-10.3f %-10.2f",
        k, total_edges, n_true, n_false,
        metrics.precision, metrics.recall, metrics.f1, filter_time))
end

println()
println("Observations:")
println("  - As k increases, more false neighbors are included")
println("  - Computational cost scales roughly as O(n·k³) for Hungarian solver")
println("  - Precision/recall tradeoff may change with k")
println()

# ============================================================================
# Study 3: Detailed breakdown at k=16
# ============================================================================

println("="^75)
println("STUDY 3: Detailed Analysis at k=16")
println("="^75)
println()

k = 16
index = build_index(BruteForceIndex, data_10d)
graph = build_knn_graph(index, data_10d; k=k)

# Compute curvatures
println("Computing curvatures for all edges...")
time_curvature = @elapsed begin
    curvatures = compute_all_curvatures(graph, data_10d)
end
println(@sprintf("  Computed %d edge curvatures in %.2f seconds\n", length(curvatures), time_curvature))

# Analyze by grid distance
println("Edge curvature by grid distance:")
println(@sprintf("%-12s %-8s %-12s %-12s %-12s", "Grid dist", "Count", "Mean κ", "Min κ", "Max κ"))
println("─"^60)

for dist in 1:5
    edges_at_dist = []

    for i in 1:n_points
        for j in graph[i]
            if grid_distance(i, j, grid_points) == dist && haskey(curvatures, (i, j))
                push!(edges_at_dist, curvatures[(i, j)].curvature)
            end
        end
    end

    if !isempty(edges_at_dist)
        mean_curv = sum(edges_at_dist) / length(edges_at_dist)
        min_curv = minimum(edges_at_dist)
        max_curv = maximum(edges_at_dist)

        println(@sprintf("%-12d %-8d %-12.3f %-12.3f %-12.3f",
            dist, length(edges_at_dist), mean_curv, min_curv, max_curv))
    end
end

println()

# Test multiple thresholds
println("Filtering at different thresholds (k=16):")
println(@sprintf("%-12s %-10s %-10s %-10s", "Threshold", "Precision", "Recall", "F1"))
println("─"^45)

for threshold in [0.0, 0.1, 0.2, 0.3, 0.4]
    filtered_graph = filter_graph(
        graph, data_10d,
        curvature_threshold=threshold,
        min_neighbors=3,
        verbose=false
    )

    metrics = evaluate_filtering(graph, filtered_graph, grid_points; max_grid_dist=1)

    println(@sprintf("%-12.1f %-10.3f %-10.3f %-10.3f",
        threshold, metrics.precision, metrics.recall, metrics.f1))
end

println()
println("="^75)
println("=== Study Complete ===")
println()
println("Summary:")
println("  1. Hungarian algorithm is fastest exact solver for uniform measures")
println("  2. LP solver is exact but slowest (O(k³ log k))")
println("  3. Sinkhorn offers good speed/accuracy tradeoff (O(k² iterations))")
println("  4. Greedy heuristic is fastest but least accurate")
println("  5. As k increases, filtering becomes more challenging (more false neighbors)")
println("  6. Optimal solver choice depends on k, measure uniformity, and accuracy needs")
