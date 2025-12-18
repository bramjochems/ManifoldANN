"""
# Example: Curvature-Based Graph Pruning for Graph-Based Indices

This example demonstrates how to:
1. Build a graph-based index (NNDescentIndex)
2. Extract its internal graph
3. Compute Ollivier-Ricci curvature
4. Prune low-curvature edges
5. Rebuild the index with the refined graph

Graph-based indices like NNDescentIndex store connectivity internally,
so we can refine them directly without full index reconstruction.
"""

using ManifoldANN
using LinearAlgebra
using Random
using Statistics
using Printf

println("="^80)
println("Curvature-Based Graph Pruning: Graph-Based Index")
println("="^80)

# Generate synthetic manifold data (Swiss roll)
Random.seed!(42)
n = 1000
t = rand(n) * 4π
h = rand(n) * 10

data = zeros(3, n)
data[1, :] = t .* cos.(t)
data[2, :] = h
data[3, :] = t .* sin.(t)

println("\nDataset: Swiss roll")
println("  Points: $n")
println("  Dimension: $(size(data, 1))")

# Step 1: Build NNDescentIndex (graph-based)
println("\n" * "-"^80)
println("Step 1: Build NNDescentIndex")
println("-"^80)

k = 20
index = build_index(NNDescentIndex, data; k=k, max_candidates=60)

println("  Index built with k=$k")
println("  Type: $(typeof(index))")

# Step 2: Extract graph from index
println("\n" * "-"^80)
println("Step 2: Extract Internal Graph")
println("-"^80)

# NNDescentIndex has an internal graph
graph = index.graph

println("  Graph nodes: $(length(graph))")
println("  k: $(graph.k)")
println("  Total edges: $(sum(length(neighbors) for neighbors in graph))")

# Step 3: Compute ORC curvatures
println("\n" * "-"^80)
println("Step 3: Compute Ollivier-Ricci Curvatures")
println("-"^80)

curvatures = compute_all_curvatures(
    graph, data;
    solver=HungarianSolver(),
    fallback_solver=NetworkSimplexSolver(),
    use_threading=true
)

curv_values = [result.curvature for result in values(curvatures)]
println("  Computed curvatures for $(length(curvatures)) edges")
@printf("  Mean κ: %.4f\n", mean(curv_values))
@printf("  Std κ:  %.4f\n", std(curv_values))
@printf("  Min κ:  %.4f\n", minimum(curv_values))
@printf("  Max κ:  %.4f\n", maximum(curv_values))

# Analyze curvature distribution
negative_count = sum(curv_values .< 0)
positive_count = sum(curv_values .>= 0)
println("\n  Curvature distribution:")
@printf("    Negative (cut across manifold): %d (%.1f%%)\n",
        negative_count, 100 * negative_count / length(curv_values))
@printf("    Positive (follow manifold):     %d (%.1f%%)\n",
        positive_count, 100 * positive_count / length(curv_values))

# Step 4: Prune low-curvature edges
println("\n" * "-"^80)
println("Step 4: Prune Low-Curvature Edges")
println("-"^80)

# Filter with κ ≥ 0 (remove negative curvature edges)
threshold = 0.0
min_neighbors = 5  # Ensure connectivity

filtered_graph = filter_graph(
    graph, data;
    curvatures=curvatures,
    curvature_threshold=threshold,
    min_neighbors=min_neighbors
)

edges_before = sum(length(neighbors) for neighbors in graph)
edges_after = sum(length(neighbors) for neighbors in filtered_graph)
edges_removed = edges_before - edges_after

println("  Threshold: κ ≥ $(threshold)")
println("  Min neighbors: $(min_neighbors)")
println("  Edges before: $(edges_before)")
println("  Edges after:  $(edges_after)")
println("  Removed:      $(edges_removed) ($(round(100 * edges_removed / edges_before, digits=1))%)")

# Step 5: Rebuild index with refined graph
println("\n" * "-"^80)
println("Step 5: Rebuild Index with Refined Graph")
println("-"^80)

# Create new NNDescentIndex using the filtered graph
# For graph-based indices, we can directly use the refined graph
refined_index = build_index(NNDescentIndex, data;
    k=filtered_graph.k,
    initial_graph=filtered_graph,  # Use filtered graph
    max_candidates=60
)

println("  ✓ Rebuilt NNDescentIndex with pruned graph")
println("  New k: $(configured_k(refined_index))")

# Step 6: Evaluate quality
println("\n" * "-"^80)
println("Step 6: Evaluate Search Quality")
println("-"^80)

# Test on random queries
n_queries = 100
query_k = 10

println("  Testing with $n_queries random queries (k=$query_k)")

# Original index
original_recalls = Float64[]
for _ in 1:n_queries
    query = data[:, rand(1:n)]

    # Ground truth from brute force
    brute = build_index(BruteForceIndex, data)
    gt_neighbors = query(brute, data, query, query_k)
    gt_ids = Set(neighbor_ids(gt_neighbors))

    # Original index
    orig_neighbors = query(index, data, query, query_k)
    orig_ids = Set(neighbor_ids(orig_neighbors))

    recall = length(intersect(gt_ids, orig_ids)) / query_k
    push!(original_recalls, recall)
end

# Refined index
refined_recalls = Float64[]
for _ in 1:n_queries
    query = data[:, rand(1:n)]

    # Ground truth
    brute = build_index(BruteForceIndex, data)
    gt_neighbors = query(brute, data, query, query_k)
    gt_ids = Set(neighbor_ids(gt_neighbors))

    # Refined index
    ref_neighbors = query(refined_index, data, query, query_k)
    ref_ids = Set(neighbor_ids(ref_neighbors))

    recall = length(intersect(gt_ids, ref_ids)) / query_k
    push!(refined_recalls, recall)
end

@printf("\n  Original index recall: %.3f (±%.3f)\n",
        mean(original_recalls), std(original_recalls))
@printf("  Refined index recall:  %.3f (±%.3f)\n",
        mean(refined_recalls), std(refined_recalls))

recall_change = mean(refined_recalls) - mean(original_recalls)
if recall_change >= 0
    @printf("  Improvement: +%.3f (%.1f%% relative)\n",
            recall_change, 100 * recall_change / mean(original_recalls))
else
    @printf("  Degradation: %.3f (%.1f%% relative)\n",
            recall_change, 100 * recall_change / mean(original_recalls))
end

# Summary
println("\n" * "="^80)
println("Summary: Graph-Based Index Pruning")
println("="^80)
println("""
Workflow:
  1. Built NNDescentIndex (graph-based) → $edges_before edges
  2. Extracted internal graph
  3. Computed ORC curvatures
  4. Pruned negative curvature edges → $edges_after edges
  5. Rebuilt index with refined graph

Results:
  • Removed $(round(100 * edges_removed / edges_before, digits=1))% of edges
  • Recall change: $(recall_change >= 0 ? "+" : "")$(round(recall_change, digits=3))
  • Graph-based indices can be refined efficiently (no full rebuild)

Benefits:
  • Removes edges that cut across manifold
  • Preserves manifold structure
  • Improves geodesic distance estimates
""")

println("="^80)
