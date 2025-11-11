#=
Example: Build a kNN graph using a KD-tree index

Run with `julia --project=. docs/examples/graphs/03-kdtree-knn-graph.jl`
=#

using ManifoldANN
using Random

rng = MersenneTwister(77)
dimension = 3
n_points = 24
k = 4

data = randn(rng, dimension, n_points)
@info "Dataset summary" dimension n_points k
println("▶ Building a KDTreeIndex and materializing a $k-nearest-neighbor graph.")

index = build_index(KDTreeIndex, data)
graph = build_knn_graph(index, data; k = k)
@info "Graph summary" n_vertices = length(graph) k = graph.k include_self = graph.include_self

println("Each vertex lists the ids of its $k nearest neighbors (approximation is exact for KD-trees).")
for (vertex, neighbors) in enumerate(graph)
    println("vertex $(lpad(vertex, 2)) -> neighbors $(join(neighbors, ", "))")
end
