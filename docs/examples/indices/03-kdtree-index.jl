#=
Example: KD-tree kNN index

Run with `julia --project=. docs/examples/indices/03-kdtree-index.jl`
=#

using ManifoldANN
using LinearAlgebra
using Random

rng = MersenneTwister(11)
dimension = 5
n_points = 64
k = 6

data = randn(rng, dimension, n_points)
println("▶ Building a KDTreeIndex with variance-based splits.")
kd_index = build_index(KDTreeIndex, data; axis_selector = :variance)
brute_index = build_index(BruteForceIndex, data)

@info "Dataset summary" dimension n_points k
@info "KD-tree metadata" axis_selector = :variance n_nodes = length(kd_index.nodes)

query_point = randn(rng, dimension)
kd_neighbors = query(kd_index, data, query_point, k)
truth_neighbors = query(brute_index, data, query_point, k)

println("Sample query vector:")
println(query_point)

function print_neighbors(title, neighbors)
    println(title)
    for neighbor in neighbors
        println(
            "  id=$(lpad(neighbor.id, 3))  dist=$(round(neighbor.dist, digits=4))",
        )
    end
end

print_neighbors("KD-tree neighbors (ids sorted by Euclidean distance):", kd_neighbors)
print_neighbors("Brute-force neighbors (ground truth):", truth_neighbors)

@info "Neighbors match" equal = (neighbor_ids(kd_neighbors) == neighbor_ids(truth_neighbors))

println("▶ Rebuilding with cyclic axis selection to demonstrate deterministic splits.")
cyclic_index = build_index(KDTreeIndex, data; axis_selector = :cyclic)
cyclic_neighbors = query(cyclic_index, data, query_point, k)
print_neighbors("Cyclic KD-tree neighbors:", cyclic_neighbors)
@info "Cyclic matches variance" equal =
    (neighbor_ids(cyclic_neighbors) == neighbor_ids(truth_neighbors))
@info "Cyclic KD-tree metadata" axis_selector = :cyclic n_nodes = length(cyclic_index.nodes)
