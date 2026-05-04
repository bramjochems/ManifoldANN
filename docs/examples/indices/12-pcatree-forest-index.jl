#=
Example: PCA-tree forest

Purpose:
    `PCATreeForestIndex` builds `n_trees` independent PCA trees with
    per-tree RNG streams. At query time, the leaf bucket each tree
    routes the query to is union'd into a single candidate set, which
    is then brute-force scanned with `index.distance`.

When to use:
    - Standard recall extension over a single PCA tree: more trees ->
      more candidates -> higher recall, at linear (and parallel) build
      cost.
    - Tree builds run on `Threads.@threads`-scheduled tasks; the forest
      is embarrassingly parallel without any user-side plumbing.
    - Deterministic at a fixed seed: per-tree RNGs are derived
      serially from the input `rng`, so results don't depend on the
      thread-scheduling order.

A critical note on splitter choice:
    PCA-tree forests only make sense with a *randomised* splitter. The
    default `pca_forest_splitter()` randomises the spectrum estimator
    (`SubsampledSVD` + `RandomizedSVD`) and the direction policy
    (`RandomTopK(3)`) so each tree sees a different partition.
    Building a forest with the deterministic `PCASplitter()` would
    silently produce N identical trees — recall would not improve
    over a single tree, and the build cost would be wasted. The
    default guards against this; user-supplied splitters should
    consume `rng` along at least one of the four trait axes.

Run with `julia --project=. docs/examples/indices/12-pcatree-forest-index.jl`
=#

using ManifoldANN
using Random

rng = MersenneTwister(13)
dimension = 16
n_points = 400
k = 10

println("=" ^ 70)
println("PCATreeForestIndex example")
println("=" ^ 70)

data = randn(rng, Float32, dimension, n_points)

# Forest of 8 trees with the randomised forest recipe. Knobs:
#   n_trees  : number of independent trees. Recall climbs with n_trees;
#              build cost is linear and parallel.
#   splitter : `pca_forest_splitter()` (randomised SVD + RandomTopK)
#              by default — see docstring for why deterministic
#              `PCASplitter()` would be wrong here.
#   rng      : seeds the per-tree RNGs. Same seed -> same forest, even
#              with different thread counts.
println("\n>> Building forest (n_trees=8, default pca_forest_splitter())")
forest = build_index(
    PCATreeForestIndex, data;
    n_trees = 8,
    rng = MersenneTwister(101),
)

# Single PCA tree with the same randomised splitter for an
# apples-to-apples comparison. The single tree only sees one leaf
# bucket; the forest unions 8 of them — different partitions cover
# different parts of the true neighbourhood.
println(">> Building single PCA tree at the same recipe")
single = build_index(
    PCATreeIndex, data;
    splitter = pca_forest_splitter(),
    rng = MersenneTwister(101),
)

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
println("   single PCA tree:                $(round(recall(single_nbrs, truth) * 100, digits = 1))%")
println("   forest (8 trees):               $(round(recall(forest_nbrs, truth) * 100, digits = 1))%")
println("   (forest typically wins; the 8 trees disagree, so the union covers more.)")

# Determinism at fixed seed: building twice with the same seed gives
# byte-identical leaf membership across all trees.
forest_a = build_index(PCATreeForestIndex, data;
                       n_trees = 8, rng = MersenneTwister(42))
forest_b = build_index(PCATreeForestIndex, data;
                       n_trees = 8, rng = MersenneTwister(42))
println("\n>> Determinism check (same seed, two builds):")
println("   query results identical: ",
        neighbor_ids(query(forest_a, data, query_vec, k)) ==
        neighbor_ids(query(forest_b, data, query_vec, k)))

# Different seeds -> different per-tree RNG streams -> different
# partitions -> typically different candidate sets and orderings.
forest_c = build_index(PCATreeForestIndex, data;
                       n_trees = 8, rng = MersenneTwister(43))
println("   different seeds give different result orderings: ",
        neighbor_ids(query(forest_a, data, query_vec, k)) !=
        neighbor_ids(query(forest_c, data, query_vec, k)))

# Build is parallel across trees: with `Threads.nthreads() >= 2` the
# 8-tree build is roughly N-way faster than serial. The example below
# is for didactic timing only — for real benchmarks see scripts/.
println("\n>> Build timings (didactic, NOT a benchmark):")
println("   Threads.nthreads() = $(Threads.nthreads())")
t_forest = @elapsed build_index(PCATreeForestIndex, data;
                                n_trees = 8, rng = MersenneTwister(7))
t_single = @elapsed build_index(PCATreeIndex, data;
                                splitter = pca_forest_splitter(),
                                rng = MersenneTwister(7))
println("   single tree:   $(round(t_single * 1e3, digits = 1)) ms")
println("   forest (8):    $(round(t_forest * 1e3, digits = 1)) ms")
println("   ratio:         $(round(t_forest / t_single, digits = 2))x  (1.0x = perfectly parallel)")

println("\n" * "=" ^ 70)
println("Walkthrough complete.")
println("=" ^ 70)
