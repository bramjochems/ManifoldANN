#=
Example: Approximate kNN graph using an HNSW index

Run with `julia --project=. docs/examples/graphs/04-hnsw-knn-graph.jl`
=#

using ManifoldANN
using Random

rng = MersenneTwister(909)
dimension = 4
n_points = 80
k = 6

data = randn(rng, dimension, n_points)

println("▶ Building an HNSW index and exporting a $k-NN graph approximation.")
index = build_index(HNSWIndex, data; M = 10, ef_construction = 100, ef_search = 60, rng = rng)
graph = build_knn_graph(index, data; k = k)

@info "Graph summary" n_vertices = length(graph) k = graph.k include_self = graph.include_self

println("Sample vertex listings (first 5 nodes):")
for vertex in 1:min(5, length(graph))
    println("vertex $(lpad(vertex, 2)) -> neighbors $(join(graph[vertex], \", \"))")
end
