#= 
Example: Brute-force kNN index

Run with `julia --project=. docs/examples/indices/01-brute-force-knn-index.jl`
=#

using ManifoldANN
using Random
using LinearAlgebra

rng = MersenneTwister(42)
dimension = 3
n_points = 12

data = randn(rng, dimension, n_points)
println("▶ Building brute-force index for $n_points points in $dimension-D space.")

index = build_index(BruteForceIndex, data)
query_point = randn(rng, dimension)
neighbor_ids = query(index, data, query_point, 4)

@info "Dataset summary" dimension n_points
println("Sample query vector:")
println(query_point)
println("Initial neighbors (ids with Euclidean distances):")
for id in neighbor_ids
    dist = LinearAlgebra.norm(@view(data[:, id]) .- query_point)
    println("  id=$(lpad(id, 3))  dist=$(round(dist, digits=4))")
end

# Insert a new point very close to the original query (remember to keep your storage up to date).
noise = 0.02 .* randn(rng, dimension)
new_point = query_point .+ noise
println("▶ Inserting one additional point that lies very close to the original query.")
data = hcat(data, new_point) # simple example; real code should manage storage efficiently
insert!(index, new_point)

println("Re-querying the original point should now surface the inserted id (", size(data, 2), ") among the top neighbors:")
neighbor_ids_after_insert = query(index, data, query_point, 4)
for id in neighbor_ids_after_insert
    dist = LinearAlgebra.norm(@view(data[:, id]) .- query_point)
    println("  id=$(lpad(id, 3))  dist=$(round(dist, digits=4))")
end
