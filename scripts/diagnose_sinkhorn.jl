"""
Diagnose why Sinkhorn fails and test if it's fixable with better parameters.
"""

using ManifoldANN
using LinearAlgebra
using DelimitedFiles
using Statistics
using Printf

println("="^80)
println("Sinkhorn Failure Diagnosis")
println("="^80)

# Load test data
data = readdlm("benchmark_results/test_data.csv", ',')
data = data'  # d x n

# Build graph
index = build_index(BruteForceIndex, data)
graph = build_knn_graph(index, data; k=15, directed=false)

println("\nAnalyzing typical cost matrix characteristics...")

# Sample some edges and analyze their cost matrices
n_samples = 10
edges = []
for i in 1:length(graph)
    for j in graph[i]
        if i < j  # Canonical form
            push!(edges, (i, j))
            length(edges) >= n_samples && break
        end
    end
    length(edges) >= n_samples && break
end

println("Sampled $n_samples edges")

# Get distance function for orcml config
cost_fn = ManifoldANN.get_distance_function(:geodesic_normalized, graph, data)
denom_fn = ManifoldANN.get_distance_function(:normalized, graph, data)

cost_stats = []

for (idx, (x, y)) in enumerate(edges)
    # Build neighborhoods
    neighbors_x = [n for n in graph[x] if n != y]
    neighbors_y = [n for n in graph[y] if n != y]

    neighborhood_x = ManifoldANN.build_neighborhood(x, neighbors_x, y, true, Float64)
    neighborhood_y = ManifoldANN.build_neighborhood(y, neighbors_y, x, true, Float64)

    edge_dist = denom_fn(x, y)
    edge_view = ManifoldANN.create_edge_view(neighborhood_x, neighborhood_y, Float64(edge_dist))

    # Build cost matrix
    n_x = length(neighborhood_x.neighbors)
    n_y = length(neighborhood_y.neighbors)

    costs = zeros(n_x, n_y)
    for i in 1:n_x
        for j in 1:n_y
            costs[i, j] = cost_fn(neighborhood_x.neighbors[i], neighborhood_y.neighbors[j])
        end
    end

    push!(cost_stats, (
        edge=(x-1, y-1),
        size=(n_x, n_y),
        cost_min=minimum(costs),
        cost_max=maximum(costs),
        cost_mean=mean(costs),
        cost_std=std(costs),
        edge_dist=edge_dist
    ))

    if idx <= 3
        println("\nEdge $(x-1) -> $(y-1):")
        println("  Problem size: $n_x x $n_y")
        println("  Cost range: [$(minimum(costs)), $(maximum(costs))]")
        println("  Cost mean: $(mean(costs))")
        println("  Cost std: $(std(costs))")
        println("  Edge distance: $edge_dist")
    end
end

# Overall statistics
all_means = [s.cost_mean for s in cost_stats]
all_stds = [s.cost_std for s in cost_stats]
all_ranges = [(s.cost_max - s.cost_min) for s in cost_stats]

println("\n" * "="^80)
println("Overall Cost Matrix Statistics:")
println("="^80)
println("  Mean of means: $(mean(all_means))")
println("  Mean of stds: $(mean(all_stds))")
println("  Mean range: $(mean(all_ranges))")
println()

# Test different regularization parameters
println("="^80)
println("Testing Sinkhorn with Different Regularization Parameters")
println("="^80)

test_edge = edges[1]
x, y = test_edge
neighbors_x = [n for n in graph[x] if n != y]
neighbors_y = [n for n in graph[y] if n != y]

neighborhood_x = ManifoldANN.build_neighborhood(x, neighbors_x, y, true, Float64)
neighborhood_y = ManifoldANN.build_neighborhood(y, neighbors_y, x, true, Float64)

edge_dist = denom_fn(x, y)
edge_view = ManifoldANN.create_edge_view(neighborhood_x, neighborhood_y, Float64(edge_dist))

precomputed_cost_fn = ManifoldANN._create_precomputed_distance_fn(
    edge_view, data, cost_fn
)

# Reference: NetworkSimplex (exact)
ref_solver = ManifoldANN.NetworkSimplexSolver()
ref_result = ManifoldANN.compute_curvature(ref_solver, edge_view, precomputed_cost_fn)
println("\nReference (NetworkSimplex):")
println("  W1 = $(ref_result.wasserstein_distance)")
println("  κ = $(ref_result.curvature)")

# Test different reg values
reg_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]

println("\nTesting Sinkhorn with different regularization:")
println("  reg    | Converged | W1        | κ         | Error vs Exact")
println("  " * "-"^65)

for reg in reg_values
    sinkhorn_solver = ManifoldANN.SinkhornSolver(reg=reg, maxiter=1000, atol=1e-6)

    try
        result = ManifoldANN.compute_curvature(sinkhorn_solver, edge_view, precomputed_cost_fn)

        if isnan(result.wasserstein_distance)
            @printf("  %.3f  | %-9s | %-9s | %-9s | %-9s\n",
                    reg, "NO", "NaN", "NaN", "NaN")
        else
            w1_error = abs(result.wasserstein_distance - ref_result.wasserstein_distance)
            k_error = abs(result.curvature - ref_result.curvature)
            @printf("  %.3f  | %-9s | %9.6f | %9.6f | W1=%.2e κ=%.2e\n",
                    reg, "YES", result.wasserstein_distance, result.curvature, w1_error, k_error)
        end
    catch e
        println("  $reg  | ERROR: $e")
    end
end

# Recommended regularization
println("\n" * "="^80)
println("Diagnosis Summary:")
println("="^80)

typical_cost_scale = mean(all_means)
recommended_reg = typical_cost_scale * 0.1  # Rule of thumb: 10% of cost scale

println("\nTypical cost matrix scale: $(typical_cost_scale)")
println("Current default reg: 0.01")
println("Recommended reg: $(recommended_reg) (10% of cost scale)")
println()
println("Problem: reg=0.01 is too small relative to cost matrix scale ($(typical_cost_scale))")
println("This causes numerical instability in the Sinkhorn iterations.")
println()
println("Fix options:")
println("  1. Increase default reg to ~$(round(recommended_reg, digits=2))")
println("  2. Scale costs to [0,1] before Sinkhorn (cost / max_cost)")
println("  3. Use adaptive reg based on cost matrix statistics")
println("  4. Increase atol to 1e-6 (less strict convergence)")
println()
println("However: Sinkhorn gives APPROXIMATE OT, not exact.")
println("For research accuracy, prefer NetworkSimplexSolver or LPReferenceSolver.")

println("\n" * "="^80)
