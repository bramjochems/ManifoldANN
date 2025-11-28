#=
Example: Validating Curvature Filtering with Ground Truth

This example demonstrates how to validate that curvature filtering correctly
removes "false neighbors" (close in Euclidean space but far on the manifold)
while preserving "true neighbors" (close on the manifold).
=#

using ManifoldANN
using LinearAlgebra
using Random

println("=== Validating Curvature Filtering ===\n")

# ============================================================================
# Generate manifold with ground truth
# ============================================================================

Random.seed!(42)
n_grid = 15
grid_points = [[Float64(i), Float64(j)] for i in 1:n_grid for j in 1:n_grid]
n_points = length(grid_points)

# Embed in 10D with nonlinear mapping
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

println("Generated $(n_points) points on a 2D grid embedded in 10D\n")

# ============================================================================
# Ground truth: geodesic distance on the grid
# ============================================================================

"""
Compute Manhattan distance on the 2D grid (proxy for geodesic distance).
"""
function grid_distance(i::Int, j::Int, grid_points)
    p1 = grid_points[i]
    p2 = grid_points[j]
    return abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])
end

"""
Check if two points are true neighbors on the grid (adjacent cells).
"""
function are_grid_neighbors(i::Int, j::Int, grid_points)
    dist = grid_distance(i, j, grid_points)
    return dist == 1  # Adjacent on grid (4-connected)
end

# ============================================================================
# Build kNN graph and compute curvatures
# ============================================================================

println("--- Building kNN Graph ---")
k = 8
index = build_index(BruteForceIndex, data_10d)
graph = build_knn_graph(index, data_10d; k=k)

println("Computing edge curvatures...")
curvatures = compute_all_curvatures(graph, data_10d)
println()

# ============================================================================
# Analyze edges: true neighbors vs false neighbors
# ============================================================================

println("--- Edge Analysis (Ground Truth) ---\n")

# Classify all edges
true_neighbor_edges = Tuple{Int,Int}[]
false_neighbor_edges = Tuple{Int,Int}[]
true_neighbor_curvatures = Float64[]
false_neighbor_curvatures = Float64[]

for i in 1:n_points
    for j in graph[i]
        edge = (i, j)
        haskey(curvatures, edge) || continue

        curv = curvatures[edge].curvature

        if are_grid_neighbors(i, j, grid_points)
            push!(true_neighbor_edges, edge)
            push!(true_neighbor_curvatures, curv)
        else
            push!(false_neighbor_edges, edge)
            push!(false_neighbor_curvatures, curv)
        end
    end
end

println("Total kNN edges: $(length(true_neighbor_edges) + length(false_neighbor_edges))")
println("True neighbors (grid-adjacent):  $(length(true_neighbor_edges))")
println("False neighbors (not adjacent):  $(length(false_neighbor_edges))")
println()

# Curvature distributions
println("Curvature Distributions:")
println("True neighbors:")
println("  Mean: $(round(sum(true_neighbor_curvatures) / length(true_neighbor_curvatures), digits=3))")
println("  Min:  $(round(minimum(true_neighbor_curvatures), digits=3))")
println("  Max:  $(round(maximum(true_neighbor_curvatures), digits=3))")
println()

println("False neighbors:")
println("  Mean: $(round(sum(false_neighbor_curvatures) / length(false_neighbor_curvatures), digits=3))")
println("  Min:  $(round(minimum(false_neighbor_curvatures), digits=3))")
println("  Max:  $(round(maximum(false_neighbor_curvatures), digits=3))")
println()

# ============================================================================
# Evaluate filtering quality
# ============================================================================

function evaluate_filtering(graph, filtered_graph, grid_points, curvatures, threshold)
    """
    Evaluate filtering by counting:
    - True positives: false neighbors removed
    - False positives: true neighbors removed
    - True negatives: true neighbors kept
    - False negatives: false neighbors kept
    """

    tp = 0  # False neighbors correctly removed
    fp = 0  # True neighbors incorrectly removed
    tn = 0  # True neighbors correctly kept
    fn = 0  # False neighbors incorrectly kept

    for i in 1:length(graph)
        original_neighbors = Set(graph[i])
        filtered_neighbors = Set(filtered_graph[i])
        removed_neighbors = setdiff(original_neighbors, filtered_neighbors)

        for j in original_neighbors
            is_true_neighbor = are_grid_neighbors(i, j, grid_points)
            was_removed = j in removed_neighbors

            if is_true_neighbor
                if was_removed
                    fp += 1  # Bad: removed a true neighbor
                else
                    tn += 1  # Good: kept a true neighbor
                end
            else
                if was_removed
                    tp += 1  # Good: removed a false neighbor
                else
                    fn += 1  # Bad: kept a false neighbor
                end
            end
        end
    end

    # Metrics
    precision = (tp + fp) > 0 ? tp / (tp + fp) : 0.0
    recall = (tp + fn) > 0 ? tp / (tp + fn) : 0.0
    f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return (
        tp = tp, fp = fp, tn = tn, fn = fn,
        precision = precision,
        recall = recall,
        f1 = f1,
        accuracy = accuracy
    )
end

println("--- Filtering Quality Evaluation ---\n")

# Test different thresholds
thresholds = [0.0, 0.1, 0.2, 0.3]

for threshold in thresholds
    println("Threshold: κ ≥ $(threshold)")

    filtered_graph = filter_graph(
        graph, data_10d,
        curvature_threshold=threshold,
        min_neighbors=3,
        verbose=false
    )

    metrics = evaluate_filtering(graph, filtered_graph, grid_points, curvatures, threshold)

    println("  Removed $(metrics.tp) false neighbors (true positives)")
    println("  Removed $(metrics.fp) true neighbors (false positives)")
    println("  Kept $(metrics.tn) true neighbors (true negatives)")
    println("  Kept $(metrics.fn) false neighbors (false negatives)")
    println("  → Precision: $(round(metrics.precision, digits=3)) (of removed edges, fraction that were false neighbors)")
    println("  → Recall:    $(round(metrics.recall, digits=3)) (of false neighbors, fraction removed)")
    println("  → F1 Score:  $(round(metrics.f1, digits=3))")
    println("  → Accuracy:  $(round(metrics.accuracy, digits=3))")
    println()
end

# ============================================================================
# Geodesic distance distortion analysis
# ============================================================================

println("--- Geodesic Distance Distortion ---\n")

"""
Compute distortion: ratio of Euclidean distance to geodesic distance.
Low distortion = distances match (true neighbor)
High distortion = Euclidean shortcut (false neighbor)
"""
function compute_distortion(i, j, data, grid_points)
    euclidean_dist = norm(data[:, i] - data[:, j])
    geodesic_dist = grid_distance(i, j, grid_points)
    return geodesic_dist == 0 ? 0.0 : euclidean_dist / geodesic_dist
end

# Sample edges and analyze
println("Sample edge analysis:")
println("(Edge type | Curvature | Geodesic dist | Euclidean dist | Distortion)")
println("─"^75)

sampled_indices = [1, 50, 100, 150, 200]
for i in sampled_indices
    i > n_points && continue
    for (idx, j) in enumerate(graph[i])
        idx > 3 && break  # Show first 3 neighbors only

        edge = (i, j)
        haskey(curvatures, edge) || continue

        curv = curvatures[edge].curvature
        geo_dist = grid_distance(i, j, grid_points)
        euc_dist = norm(data_10d[:, i] - data_10d[:, j])
        distortion = geo_dist == 0 ? 0.0 : euc_dist / geo_dist

        edge_type = are_grid_neighbors(i, j, grid_points) ? "TRUE " : "FALSE"

        println("$(edge_type) | κ=$(round(curv, digits=3)) | geo=$(geo_dist) | euc=$(round(euc_dist, digits=2)) | dist=$(round(distortion, digits=3))")
    end
end

println("\n=== Validation Complete ===")
println("\nInterpretation:")
println("- High curvature (κ > 0.3) → edges follow manifold (true neighbors)")
println("- Low curvature (κ < 0.0) → edges cut across manifold (false neighbors)")
println("- Good filtering should have high precision (removed edges are false neighbors)")
println("  and high recall (most false neighbors removed)")
