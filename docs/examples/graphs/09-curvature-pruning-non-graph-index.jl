"""
# Example: Curvature-Based Graph Pruning for Non-Graph Indices

This example demonstrates how to:
1. Build a non-graph-based index (HNSW, BruteForce, KDTree, etc.)
2. Extract a k-NN graph from the index
3. Compute Ollivier-Ricci curvature
4. Prune low-curvature edges
5. Rebuild a new index using only the refined edges

For non-graph indices, we cannot modify the index directly, so we:
- Extract connectivity → Prune → Build new graph-based index
"""

using ManifoldANN
using LinearAlgebra
using Random
using Statistics
using Printf

println("="^80)
println("Curvature-Based Graph Pruning: Non-Graph Index")
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

# Step 1: Build non-graph-based index (HNSW)
println("\n" * "-"^80)
println("Step 1: Build HNSW Index (Non-Graph Based)")
println("-"^80)

k = 20
M = 16  # HNSW connectivity parameter

index = build_index(HNSWIndex, data; M=M, ef_construction=200)

println("  Index built: HNSWIndex")
println("  M: $M")
println("  Type: $(typeof(index))")
println("\n  Note: HNSW has internal connectivity, but it's hierarchical")
println("        and not directly modifiable like NNDescentIndex")

# Step 2: Extract k-NN graph
println("\n" * "-"^80)
println("Step 2: Extract k-NN Graph from Index")
println("-"^80)

# Query the index to build a k-NN graph
graph = build_knn_graph(index, data; k=k)

println("  Graph extracted via querying")
println("  Nodes: $(length(graph))")
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

# Analyze distribution
negative_count = sum(curv_values .< 0)
println("\n  Curvature distribution:")
@printf("    Negative (cut across): %d (%.1f%%)\n",
        negative_count, 100 * negative_count / length(curv_values))
@printf("    Positive (follow manifold): %d (%.1f%%)\n",
        length(curv_values) - negative_count,
        100 * (length(curv_values) - negative_count) / length(curv_values))

# Step 4: Prune low-curvature edges
println("\n" * "-"^80)
println("Step 4: Prune Low-Curvature Edges")
println("-"^80)

threshold = 0.0
min_neighbors = 5

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
println("  Edges before: $(edges_before)")
println("  Edges after:  $(edges_after)")
println("  Removed:      $(edges_removed) ($(round(100 * edges_removed / edges_before, digits=1))%)")

# Step 5: Build new index from refined graph
println("\n" * "-"^80)
println("Step 5: Build New Index from Refined Graph")
println("-"^80)

println("  Strategy: Build NNDescentIndex using filtered graph")
println("  Rationale: Graph-based index can use refined connectivity directly")

refined_index = build_index(NNDescentIndex, data;
    k=filtered_graph.k,
    initial_graph=filtered_graph,
    max_candidates=60
)

println("  ✓ Built NNDescentIndex with refined graph")
println("  Type: $(typeof(refined_index))")
println("  k: $(configured_k(refined_index))")

# Alternative: Could rebuild HNSW, but it wouldn't use the pruned graph directly
println("\n  Alternative (not shown):")
println("    • Rebuild HNSW from scratch (ignores pruned graph)")
println("    • Use filtered graph as initialization for iterative refinement")

# Step 6: Compare search quality
println("\n" * "-"^80)
println("Step 6: Evaluate Search Quality")
println("-"^80)

n_queries = 100
query_k = 10

println("  Testing with $n_queries random queries (k=$query_k)")

# Original HNSW
println("\n  Original HNSW...")
hnsw_recalls = Float64[]
for _ in 1:n_queries
    query_point = data[:, rand(1:n)]

    # Ground truth
    brute = build_index(BruteForceIndex, data)
    gt_neighbors = query(brute, data, query_point, query_k)
    gt_ids = Set(neighbor_ids(gt_neighbors))

    # HNSW
    hnsw_neighbors = query(index, data, query_point, query_k; ef=50)
    hnsw_ids = Set(neighbor_ids(hnsw_neighbors))

    recall = length(intersect(gt_ids, hnsw_ids)) / query_k
    push!(hnsw_recalls, recall)
end

# Refined NNDescent
println("  Refined NNDescent...")
refined_recalls = Float64[]
for _ in 1:n_queries
    query_point = data[:, rand(1:n)]

    # Ground truth
    brute = build_index(BruteForceIndex, data)
    gt_neighbors = query(brute, data, query_point, query_k)
    gt_ids = Set(neighbor_ids(gt_neighbors))

    # Refined
    ref_neighbors = query(refined_index, data, query_point, query_k)
    ref_ids = Set(neighbor_ids(ref_neighbors))

    recall = length(intersect(gt_ids, ref_ids)) / query_k
    push!(refined_recalls, recall)
end

@printf("\n  Original HNSW recall:  %.3f (±%.3f)\n",
        mean(hnsw_recalls), std(hnsw_recalls))
@printf("  Refined NNDescent recall: %.3f (±%.3f)\n",
        mean(refined_recalls), std(refined_recalls))

# Step 7: Show workflow comparison
println("\n" * "="^80)
println("Summary: Non-Graph Index Pruning Workflow")
println("="^80)

println("""
Workflow for Non-Graph Indices (HNSW, KDTree, LSH):

  1. Build initial index (HNSW, KDTree, etc.)
     └─> Index with internal structure (not directly modifiable)

  2. Extract k-NN graph by querying index
     └─> build_knn_graph(index, data; k=20)

  3. Compute ORC curvatures on extracted graph
     └─> compute_all_curvatures(graph, data)

  4. Filter low-curvature edges
     └─> filter_graph(graph, data; threshold=0.0)

  5. Build NEW index from refined graph
     ├─> Option A: NNDescentIndex (uses graph directly)
     ├─> Option B: Rebuild original index type (full rebuild)
     └─> Option C: Use as initialization for iterative refinement

Results:
  • Removed $(round(100 * edges_removed / edges_before, digits=1))% of edges
  • Original HNSW: $(round(mean(hnsw_recalls), digits=3)) recall
  • Refined NNDescent: $(round(mean(refined_recalls), digits=3)) recall

Key Differences from Graph-Based Indices:
  • Cannot modify HNSW/KDTree/LSH in-place
  • Must extract graph → filter → rebuild
  • Best to switch to NNDescentIndex for refined graph
  • Alternative: Use filtered edges to guide new index construction
""")

println("="^80)
