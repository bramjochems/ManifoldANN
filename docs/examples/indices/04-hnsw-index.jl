#=
Example: HNSW index for approximate kNN search

Run with `julia --project=. docs/examples/indices/04-hnsw-index.jl`
=#

using ManifoldANN
using Random
using LinearAlgebra

rng = MersenneTwister(314)
dimension = 5
n_points = 128
k = 8

data = randn(rng, dimension, n_points)
println("▶ Building an HNSW index (M=12, ef_construction=120, ef_search=80).")
hnsw = build_index(
    HNSWIndex,
    data;
    M = 12,
    ef_construction = 120,
    ef_search = 80,
    rng = rng,
)
brute = build_index(BruteForceIndex, data)

@info "Dataset summary" dimension n_points k

query_point = randn(rng, dimension)
approx_neighbors = query(hnsw, data, query_point, k; ef_search = 100)
truth_neighbors = query(brute, data, query_point, k)

function print_neighbors(label, neighbors)
    println(label)
    for neighbor in neighbors
        println(
            "  id=$(lpad(neighbor.id, 3)) dist=$(round(neighbor.dist, digits=4))",
        )
    end
end

print_neighbors("HNSW neighbors:", approx_neighbors)
print_neighbors("Brute-force neighbors:", truth_neighbors)

overlap =
    length(
        intersect(neighbor_ids(approx_neighbors) |> Set, neighbor_ids(truth_neighbors) |> Set),
    ) / k
@info "Recall for this query" recall = overlap

println("▶ Demonstrating insert! with a nearby point.")
new_point = query_point .+ 0.05 .* randn(rng, dimension)
data = hcat(data, new_point)
insert!(hnsw, data, new_point; rng = rng)
println("Re-querying after insertion (expect new id $(size(data, 2)) to appear).")
approx_after = query(hnsw, data, query_point, k; ef_search = 100)
print_neighbors("HNSW neighbors after insert:", approx_after)
