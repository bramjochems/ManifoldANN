#=
Example: Build a kNN graph from any ANN index

Run with `julia --project=. docs/examples/graphs/01-build-knn-graph.jl`
=#

using ManifoldANN
using Random

rng = MersenneTwister(13)
dimension = 2
n_points = 6
k = 2

data = randn(rng, dimension, n_points)
labels = ["point_$(i)" for i in 1:n_points]
@info "Dataset summary" dimension n_points k
println("▶ Building brute-force index and materializing a $k-NN graph (labels as metadata).")

# BruteForceIndex is handy for demos; any other ANN index works as long as it
# implements the `query` interface.
index = build_index(BruteForceIndex, data)

graph = build_knn_graph(index, data; k = k, metadata = labels)

@info "Graph summary" n_vertices = length(graph) k = graph.k include_self = graph.include_self carries_metadata = has_metadata(graph)

println("Each vertex lists ids, neighbor labels, and coordinates for the $k nearest neighbors.")
for (vertex, neighbors) in enumerate(graph)
    neighbor_labels = [labels[n] for n in neighbors]
    @info "Neighbors" vertex_label = node_metadata(graph, vertex) neighbors = neighbors neighbor_labels = neighbor_labels points = map(n -> data[:, n], neighbors)
end
