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
@info "Dataset summary" dimension n_points k
println("▶ Building brute-force index and materializing a $k-NN graph.")

# BruteForceIndex is handy for demos; any other ANN index works as long as it
# implements the `query` interface.
index = build_index(BruteForceIndex, data)

graph = build_knn_graph(index, data; k = k)

@info "Graph summary" n_vertices = length(graph) k = graph.k include_self = graph.include_self

println("Each vertex is listed with the ids (and coordinates) of its $k nearest neighbors.")
for (vertex, neighbors) in enumerate(graph)
    @info "Neighbors" vertex neighbors points = map(n -> data[:, n], neighbors)
end
