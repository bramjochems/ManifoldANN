#=
Example: Random-projection tree (single tree)

Purpose:
    A single random-projection tree partitions the dataset by repeatedly
    splitting on a random hyperplane defined by two sampled points
    (`TwoPointSplitter`, the default). At query time the tree routes the
    query down to one leaf bucket and brute-forces it.

When to use:
    - Cheap, dependency-free baseline tree for ANN.
    - Building block for `RPTreeForestIndex` (see 10-rptree-forest-index.jl).
    - A single tree's recall is intrinsically limited; if you care about
      recall, prefer the forest variant.

Run with `julia --project=. docs/examples/indices/09-rptree-index.jl`
=#

using ManifoldANN
using Random

rng = MersenneTwister(7)
dimension = 16
n_points = 200
k = 10

println("=" ^ 70)
println("Single RPTreeIndex example")
println("=" ^ 70)

# Small synthetic dataset: standard-normal Float32 columns.
data = randn(rng, Float32, dimension, n_points)

# Build the tree. Knobs:
#   leaf_cap : maximum bucket size at a leaf (default 32). Larger leaves
#              -> better recall but slower queries.
#   splitter : an `AbstractRPSplitter`. Default is the two-point splitter
#              (`ManifoldANN.TwoPointSplitter()`), which picks a random
#              hyperplane through two sampled points. Users can subtype
#              `AbstractRPSplitter` to plug in custom split strategies;
#              see the `AbstractRPSplitter` docstring in src/utils/rptree.jl.
#   rng      : controls the splitter's random sampling so builds are
#              reproducible at fixed seed.
println("\n>> Building RPTreeIndex (leaf_cap=32, default TwoPointSplitter)")
rp_index = build_index(RPTreeIndex, data; leaf_cap = 32, rng = MersenneTwister(1))

# Brute-force baseline for ground-truth comparison.
brute = build_index(BruteForceIndex, data)

# Query a held-out random point.
query_vec = randn(rng, Float32, dimension)
rp_neighbors = query(rp_index, data, query_vec, k)
truth = query(brute, data, query_vec, k)

# Recall sanity check. A single RP tree typically lands somewhere below
# 100% recall on random Gaussian data — that's expected, not a bug.
function recall(approx, truth)
    return length(intersect(Set(neighbor_ids(approx)), Set(neighbor_ids(truth)))) /
           length(truth)
end

r = recall(rp_neighbors, truth)
println("\n>> Recall@$k vs brute force: $(round(r * 100, digits = 1))%")
println("   (single RP tree is intentionally limited; use the forest for higher recall)")

# Show top-3 neighbors side-by-side.
println("\n>> Top-3 RP-tree neighbors:")
for nbr in rp_neighbors[1:min(3, length(rp_neighbors))]
    println("   id=$(lpad(nbr.id, 4))  dist=$(round(nbr.dist, digits = 4))")
end
println(">> Top-3 brute-force neighbors:")
for nbr in truth[1:3]
    println("   id=$(lpad(nbr.id, 4))  dist=$(round(nbr.dist, digits = 4))")
end

println("\n>> See 10-rptree-forest-index.jl for the recall extension via")
println("   independent trees + union-of-leaves candidate scanning.")
