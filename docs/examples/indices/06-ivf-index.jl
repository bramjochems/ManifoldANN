#=
Example: IVF (Inverted File) index with KMeans clustering

This example demonstrates the basic IVF pattern: coarse clustering with KMeans,
followed by refined search within the nearest clusters using HNSW.

Run with `julia --project=. docs/examples/indices/06-ivf-index.jl`
=#

using ManifoldANN
using Distances
using Random
using LinearAlgebra

rng = MersenneTwister(42)
dimension = 64
n_points = 10_000
k = 10  # Number of neighbors to find
n_clusters = 50  # Number of coarse clusters
n_probe = 5  # Number of clusters to probe during query

println("=" ^ 70)
println("IVF (Inverted File) Index Example")
println("=" ^ 70)
println()

# Generate synthetic data
println("▶ Generating random dataset ($n_points points, $dimension dimensions)")
data = randn(rng, Float32, dimension, n_points)
println()

# Build IVF index
println("▶ Building IVF index")
println("  Configuration:")
println("    - Clusters: $n_clusters")
println("    - Probe: $n_probe nearest clusters")
println("    - Terminal index: HNSW (M=16)")

config = TransformedConfig(
    KMeansTransform(
        k=n_clusters,
        distance=Euclidean(),
        init=:kmeans_plus_plus,  # Better initialization than random
        max_iters=50,
        tol=1e-6
    ),
    TopKRouting(n_probe),
    TerminalConfig(HNSWIndex, (M=16, ef_construction=200, ef_search=50))
)

@time ivf_index = build_index(MultiLevelIndex, data, config)
println()

# Build baseline indices for comparison
println("▶ Building baseline indices for comparison")
@time hnsw_index = build_index(HNSWIndex, data; M=16, ef_construction=200, ef_search=50)
@time brute_index = build_index(BruteForceIndex, data)
println()

# Query the indices
query_point = randn(rng, Float32, dimension)
println("▶ Querying for $k nearest neighbors")
println()

println("  IVF index:")
@time ivf_neighbors = query(ivf_index, data, query_point, k)

println("  HNSW baseline:")
@time hnsw_neighbors = query(hnsw_index, data, query_point, k)

println("  Brute-force (ground truth):")
@time truth_neighbors = query(brute_index, data, query_point, k)
println()

# Compute recalls
function compute_recall(approx_ids, truth_ids)
    return length(intersect(Set(approx_ids), Set(truth_ids))) / length(truth_ids)
end

ivf_recall = compute_recall(ivf_neighbors, truth_neighbors)
hnsw_recall = compute_recall(hnsw_neighbors, truth_neighbors)

println("▶ Results:")
println("  IVF recall:  $(round(ivf_recall * 100, digits=1))%")
println("  HNSW recall: $(round(hnsw_recall * 100, digits=1))%")
println()

# Show actual distances for verification
function print_neighbors(label, ids, n_show=5)
    println("$label (showing first $n_show):")
    for (i, id) in enumerate(ids[1:min(n_show, length(ids))])
        dist = norm(@view(data[:, id]) - query_point)
        println("  $i. id=$(lpad(id, 5))  dist=$(round(dist, digits=4))")
    end
    println()
end

print_neighbors("IVF neighbors", ivf_neighbors)
print_neighbors("Ground truth", truth_neighbors)

# Additional insights
println("▶ Index structure:")
println("  Root transform: KMeansTransform with $n_clusters centroids")
println("  Number of child indices: $(length(ivf_index.root.indices))")
println("  Each child is: $(typeof(ivf_index.root.indices[1]))")
println()

println("=" ^ 70)
println("Key takeaways:")
println("- IVF provides fast approximate search by pruning search space")
println("- Trade-off: n_probe controls recall vs speed")
println("- Higher n_probe → higher recall but slower queries")
println("- KMeans++ initialization improves cluster quality")
println("=" ^ 70)
