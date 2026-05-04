#=
Example: Random-projection tree forest

Purpose:
    `RPTreeForestIndex` builds `n_trees` independent random-projection
    trees with per-tree RNG streams. At query time, the leaf bucket each
    tree routes the query to is union'd into a single candidate set,
    which is then brute-force scanned with `index.distance`.

When to use:
    - Standard recall extension over a single RP tree: more trees ->
      more candidates -> higher recall, at linear build cost.
    - Tree builds run on `Threads.@spawn`-ed tasks; the forest is
      embarrassingly parallel without any user-side plumbing.
    - Deterministic at a fixed seed: per-tree RNGs are derived
      serially from the input `rng`, so results don't depend on the
      thread-scheduling order.

Run with `julia --project=. docs/examples/indices/10-rptree-forest-index.jl`
=#

using ManifoldANN
using Random

rng = MersenneTwister(7)
dimension = 16
n_points = 200
k = 10

println("=" ^ 70)
println("RPTreeForestIndex example")
println("=" ^ 70)

data = randn(rng, Float32, dimension, n_points)

# Forest of 8 trees, each with leaf_cap = 16. Knobs:
#   n_trees  : number of independent trees. Recall climbs with n_trees;
#              build cost is linear and parallel.
#   leaf_cap : per-tree leaf bucket size. Forest candidate-set size is
#              roughly bounded by `n_trees * leaf_cap` (minus overlap).
#   rng      : seeds the per-tree RNGs. Same seed -> same forest, even
#              with different thread counts.
println("\n>> Building forest (n_trees=8, leaf_cap=16)")
forest = build_index(
    RPTreeForestIndex, data;
    n_trees = 8, leaf_cap = 16,
    rng = MersenneTwister(101),
)

# Single tree at MATCHED candidate budget for an apples-to-apples comparison:
# the forest can see up to 8 * 16 = 128 candidates, so we give the single
# tree leaf_cap = 128. Same total scan work, but forest's candidates come
# from 8 independent partitions instead of one.
println(">> Building single tree at matched candidate budget (leaf_cap=128)")
single = build_index(RPTreeIndex, data; leaf_cap = 128, rng = MersenneTwister(101))

# Ground truth.
brute = build_index(BruteForceIndex, data)

query_vec = randn(rng, Float32, dimension)
forest_nbrs = query(forest, data, query_vec, k)
single_nbrs = query(single, data, query_vec, k)
truth = query(brute, data, query_vec, k)

function recall(approx, truth)
    return length(intersect(Set(neighbor_ids(approx)), Set(neighbor_ids(truth)))) /
           length(truth)
end

println("\n>> Recall@$k vs brute force:")
println("   single tree (leaf_cap=128):    $(round(recall(single_nbrs, truth) * 100, digits = 1))%")
println("   forest (8 trees, leaf_cap=16): $(round(recall(forest_nbrs, truth) * 100, digits = 1))%")
println("   (forest typically wins at matched budget; the trees disagree, so the union covers more.)")

# Determinism at fixed seed: building twice with the same seed gives
# byte-identical leaf membership.
forest_a = build_index(RPTreeForestIndex, data; n_trees = 8, leaf_cap = 16,
                       rng = MersenneTwister(42))
forest_b = build_index(RPTreeForestIndex, data; n_trees = 8, leaf_cap = 16,
                       rng = MersenneTwister(42))
println("\n>> Determinism check (same seed, two builds):")
println("   query results identical: ",
        neighbor_ids(query(forest_a, data, query_vec, k)) ==
        neighbor_ids(query(forest_b, data, query_vec, k)))

# Different seeds -> different forests -> typically different candidate sets.
forest_c = build_index(RPTreeForestIndex, data; n_trees = 8, leaf_cap = 16,
                       rng = MersenneTwister(43))
println("   different seeds give different result orderings: ",
        neighbor_ids(query(forest_a, data, query_vec, k)) !=
        neighbor_ids(query(forest_c, data, query_vec, k)))
