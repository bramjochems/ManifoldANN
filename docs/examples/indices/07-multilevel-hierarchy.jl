#=
Example: Multi-level hierarchical index with nested KMeans

This example demonstrates a two-level hierarchy:
  Level 1: Coarse KMeans clustering (e.g., 20 clusters)
  Level 2: Fine KMeans clustering within each coarse cluster (e.g., 10 clusters)
  Level 3: HNSW index within each fine cluster

This creates a tree structure that progressively narrows the search space.

Run with `julia --project=. docs/examples/indices/07-multilevel-hierarchy.jl`
=#

using ManifoldANN
using Distances
using Random
using LinearAlgebra

rng = MersenneTwister(123)
dimension = 32
n_points = 5_000
k = 10

# Hierarchy parameters
n_coarse_clusters = 20   # Level 1: coarse partitioning
n_fine_clusters = 10     # Level 2: fine partitioning within each coarse cluster
n_probe_coarse = 3       # Probe 3 coarse clusters
n_probe_fine = 2         # Probe 2 fine clusters within each coarse cluster
# Total fine clusters probed: 3 × 2 = 6 out of 20 × 10 = 200

println("=" ^ 70)
println("Multi-Level Hierarchical Index Example")
println("=" ^ 70)
println()

# Generate data
println("▶ Generating dataset ($n_points points, $dimension dimensions)")
data = randn(rng, Float32, dimension, n_points)
println()

# Build multi-level index
println("▶ Building two-level hierarchical index")
println("  Level 1 (coarse): $n_coarse_clusters clusters, probe $n_probe_coarse")
println("  Level 2 (fine):   $n_fine_clusters clusters per coarse, probe $n_probe_fine")
println("  Level 3 (terminal): HNSW indices")
println("  Total structure: $(n_coarse_clusters * n_fine_clusters) HNSW indices")
println()

multilevel_config = TransformedConfig(
    # Level 1: Coarse clustering
    KMeansTransform(
        k=n_coarse_clusters,
        distance=Euclidean(),
        init=:kmeans_plus_plus
    ),
    TopKRouting(n_probe_coarse),
    # Level 2: Fine clustering (nested config)
    TransformedConfig(
        KMeansTransform(
            k=n_fine_clusters,
            distance=Euclidean(),
            init=:random  # Faster for small clusters
        ),
        TopKRouting(n_probe_fine),
        # Level 3: Terminal HNSW
        TerminalConfig(HNSWIndex, (M=8, ef_construction=100))
    )
)

@time multilevel_index = build_index(MultiLevelIndex, data, multilevel_config)
println()

# Build single-level IVF for comparison
println("▶ Building single-level IVF (for comparison)")
println("  Single level: $n_coarse_clusters clusters, probe $n_probe_coarse")

ivf_config = TransformedConfig(
    KMeansTransform(
        k=n_coarse_clusters,
        distance=Euclidean(),
        init=:kmeans_plus_plus
    ),
    TopKRouting(n_probe_coarse),
    TerminalConfig(HNSWIndex, (M=8, ef_construction=100))
)

@time ivf_index = build_index(MultiLevelIndex, data, ivf_config)
println()

# Build brute-force baseline
println("▶ Building brute-force baseline")
@time brute_index = build_index(BruteForceIndex, data)
println()

# Query all indices
query_point = randn(rng, Float32, dimension)
println("▶ Querying for $k nearest neighbors")
println()

println("  Multi-level index:")
@time multilevel_neighbors = query(multilevel_index, data, query_point, k)

println("  Single-level IVF:")
@time ivf_neighbors = query(ivf_index, data, query_point, k)

println("  Brute-force:")
@time truth_neighbors = query(brute_index, data, query_point, k)
println()

# Compute recalls
function compute_recall(approx_ids, truth_ids)
    return length(intersect(Set(approx_ids), Set(truth_ids))) / length(truth_ids)
end

multilevel_recall = compute_recall(multilevel_neighbors, truth_neighbors)
ivf_recall = compute_recall(ivf_neighbors, truth_neighbors)

println("▶ Recall comparison:")
println("  Multi-level: $(round(multilevel_recall * 100, digits=1))%")
println("  Single-level: $(round(ivf_recall * 100, digits=1))%")
println()

# Inspect the structure
println("▶ Index structure analysis:")
println()
println("Multi-level index:")
println("  Root level:")
println("    Transform: $(typeof(multilevel_index.root.transform))")
println("    Number of children: $(length(multilevel_index.root.indices))")
println("  Second level (first coarse cluster):")
first_child = multilevel_index.root.indices[1]
println("    Transform: $(typeof(first_child.transform))")
println("    Number of children: $(length(first_child.indices))")
println("  Third level (first fine cluster):")
first_grandchild = first_child.indices[1]
println("    Terminal index: $(typeof(first_grandchild))")
println()

println("Single-level IVF:")
println("  Root level:")
println("    Transform: $(typeof(ivf_index.root.transform))")
println("    Number of children: $(length(ivf_index.root.indices))")
println("  Second level:")
println("    Terminal indices: $(typeof(ivf_index.root.indices[1]))")
println()

println("=" ^ 70)
println("Key insights:")
println()
println("1. Multi-level hierarchies enable progressive search space pruning")
println("   - First level: narrow to ~$n_probe_coarse coarse clusters")
println("   - Second level: further narrow within those clusters")
println()
println("2. Trade-offs:")
println("   - More levels: finer-grained search, more overhead")
println("   - Fewer levels: simpler structure, less pruning")
println()
println("3. When to use multi-level:")
println("   - Very large datasets (10M+ points)")
println("   - When single-level IVF becomes memory/time bottleneck")
println("   - Research on hierarchical partitioning strategies")
println()
println("4. This pattern is unique to ManifoldANN (not in standard FAISS)")
println("=" ^ 70)
