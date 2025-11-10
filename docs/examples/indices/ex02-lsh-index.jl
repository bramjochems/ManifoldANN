#=
Example: LSH index with inserts

Run with `julia --project=. docs/examples/indices/ex02-lsh-index.jl`
=#

using ManifoldANN
using Random
using LinearAlgebra

rng = MersenneTwister(7)
dimension = 6
n_points = 200

data = randn(rng, dimension, n_points)
@info "Dataset summary" dimension n_points
println("▶ Building two LSH indices: random-hyperplane (angular) and binning (L2).")

hyperplane_index = build_index(
    LSHIndex,
    data;
    n_tables = 10,
    hash_length = 12,
    rng = rng,
    hash_factory = make_random_hyperplane_hash,
)

query_point = randn(rng, dimension)
neighbor_ids = query(hyperplane_index, data, query_point, 3)
println("Random-hyperplane sample query results (ids): ", neighbor_ids)

# Insert a new point and keep storage aligned.
new_point = randn(rng, dimension)
println("▶ Demonstrating insert! by appending a single point to the dataset.")
data = hcat(data, new_point)
insert!(hyperplane_index, new_point)

neighbor_ids_after_insert = query(hyperplane_index, data, new_point, 1)
println("Nearest neighbor for inserted point (should return its own id): ", neighbor_ids_after_insert)

# Demonstrate binning hash for Euclidean distance.
println("▶ Rebuilding with a binning hash family suited for Euclidean distances.")
binning_index = build_index(
    LSHIndex,
    data;
    n_tables = 8,
    hash_length = 6,
    rng = rng,
    hash_factory = make_binning_hash,
    bin_width = 2.5,
    use_offset = true,
)

euclid_ids = query(binning_index, data, query_point, 3)
println("Binning-hash neighbors (ids): ", isempty(euclid_ids) ? "(none; consider raising bin_width or lowering hash_length)" : string(euclid_ids))

# Compare recall against brute-force for a handful of random queries.
brute = build_index(BruteForceIndex, data)
k = 5
n_eval = 5
queries = [randn(rng, dimension) for _ in 1:n_eval]
println("▶ Evaluating recall on $n_eval random queries relative to brute-force results.")

function neighbors_for(index, queries)
    [query(index, data, q, k) for q in queries]
end

true_neighbors = [query(brute, data, q, k) for q in queries]
hyperplane_neighbors = neighbors_for(hyperplane_index, queries)
binning_neighbors = neighbors_for(binning_index, queries)

recall_hyperplane = recall_at_k(hyperplane_neighbors, true_neighbors)
recall_binning = recall_at_k(binning_neighbors, true_neighbors)

@info "Recall over $n_eval queries" hyperplane = recall_hyperplane binning = recall_binning

println("Shown below: for each query we list the approximate neighbors vs the brute-force ground truth.")
for i in 1:n_eval
    q = queries[i]
    println("Query $i (norm=$(round(LinearAlgebra.norm(q), digits=3))):")
    sorted_truth = sort(true_neighbors[i]; by = id -> LinearAlgebra.norm(@view(data[:, id]) .- q))
    sorted_hyper = sort(hyperplane_neighbors[i]; by = id -> LinearAlgebra.norm(@view(data[:, id]) .- q))
    sorted_bin = sort(binning_neighbors[i]; by = id -> LinearAlgebra.norm(@view(data[:, id]) .- q))
    println("  truth      -> ", sorted_truth)
    println("  hyperplane -> ", sorted_hyper)
    println("  binning    -> ", sorted_bin)
end
