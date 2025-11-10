#=
Example: Build a kNN graph using an LSH index

Run with `julia --project=. docs/examples/graphs/02-lsh-knn-graph.jl`
=#

using ManifoldANN
using Random

rng = MersenneTwister(2024)
dimension = 3
n_clusters = 5
points_per_cluster = 4
n_points = n_clusters * points_per_cluster
k = 3

function clustered_dataset(rng, dimension, n_clusters, points_per_cluster)
    total = n_clusters * points_per_cluster
    data = Array{Float64}(undef, dimension, total)
    idx = 1
    for _ in 1:n_clusters
        center = randn(rng, dimension)
        for _ in 1:points_per_cluster
            data[:, idx] = center .+ 0.15 * randn(rng, dimension)
            idx += 1
        end
    end
    return data
end

data = clustered_dataset(rng, dimension, n_clusters, points_per_cluster)

@info "Dataset summary" dimension n_points k n_clusters points_per_cluster
println("▶ Building an LSH index (binning hash) and exporting its connectivity as a kNN graph.")

index = build_index(
    LSHIndex,
    data;
    n_tables = 10,
    hash_length = 3,
    rng = rng,
    hash_factory = make_binning_hash,
    bin_width = 1.2,
    use_offset = true,
)

graph = build_knn_graph(index, data; k = k)
@info "LSH-derived graph" vertices = length(graph) k = graph.k include_self = graph.include_self

println("Each vertex is listed with the ids (and coordinates) returned by the approximate search.")
for (vertex, neighbors) in enumerate(graph)
    @info "Neighbors" vertex neighbors points = map(n -> data[:, n], neighbors)
end
