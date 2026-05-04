#=
Example: PCATreeIndex with the four-trait swappable policy surface

Purpose:
    `PCATreeIndex` is a binary partition tree that splits each node along
    a direction derived from the local PCA spectrum. Unlike most ANN
    libraries, the per-node behaviour is exposed as FOUR orthogonal
    extension points, composed by `PCASplitter`:

        AbstractSpectrumEstimator     - how to get principal directions
        AbstractSplitDirectionPolicy  - which direction to split on
        AbstractStoppingCriterion     - when to stop recursing
        AbstractSplitValuePolicy      - where along the direction to cut

    Each axis is independent. This file walks through swapping each one.

When to use:
    - When you want a structurally adaptive tree that follows the data's
      principal directions (vs the axis-aligned KD-tree or random-hyperplane
      RP tree).
    - When you want to randomise specific axes (e.g. direction, split
      value) for forest ensembles while keeping others deterministic.

Run with `julia --project=. docs/examples/indices/11-pcatree-index.jl`
=#

using ManifoldANN
using Random
using LinearAlgebra

rng = MersenneTwister(2026)
dimension = 16
n_points = 200
k = 10

println("=" ^ 70)
println("PCATreeIndex: swappable-policy walkthrough")
println("=" ^ 70)

data = randn(rng, Float32, dimension, n_points)
brute = build_index(BruteForceIndex, data)
query_vec = randn(rng, Float32, dimension)
truth = query(brute, data, query_vec, k)

function recall(approx, truth)
    return length(intersect(Set(neighbor_ids(approx)), Set(neighbor_ids(truth)))) /
           length(truth)
end

# --------------------------------------------------------------------------
# 1. Default `PCASplitter()` — the deterministic baseline.
# --------------------------------------------------------------------------
# Defaults are:
#   - ExactSVD                          (gold-standard spectrum)
#   - TopComponent                      (split on the leading PC)
#   - AnyOf(MaxLeafSize(32),
#           IntrinsicDimRatio(0.1, n_floor=256))   (composite stopping)
#   - MedianSplit                       (balanced tree)
println("\n[1] Default PCASplitter — deterministic baseline")
default_index = build_index(PCATreeIndex, data; rng = MersenneTwister(1))
nbrs = query(default_index, data, query_vec, k)
println("    recall@$k = $(round(recall(nbrs, truth) * 100, digits = 1))%")
println("    n_nodes   = $(length(default_index.nodes))")

# --------------------------------------------------------------------------
# 2. Spectrum-estimator swap: ExactSVD -> RandomizedSVD.
# --------------------------------------------------------------------------
# `RandomizedSVD(rank, n_iter=2)` uses Halko-Martinsson-Tropp sketching:
# only the leading `rank` directions are computed, in O(d * n * (rank+p))
# instead of full O(min(d,n) * d * n). On large `d` this is the
# difference between a feasible build and an unaffordable one. The split
# direction is still meaningful — just sampled rather than exact.
println("\n[2] Spectrum estimator swap: ExactSVD -> RandomizedSVD")
rsvd_splitter = PCASplitter(
    RandomizedSVD(10; oversample = 10, n_iter = 2),  # only top-10 directions
    TopComponent(),
    AnyOf(MaxLeafSize(32), IntrinsicDimRatio(0.1; n_floor = 256)),
    MedianSplit(),
)
rsvd_index = build_index(PCATreeIndex, data; splitter = rsvd_splitter,
                         rng = MersenneTwister(1))
println("    recall@$k = $(round(recall(query(rsvd_index, data, query_vec, k), truth) * 100, digits = 1))%")
println("    The build still routes the same way; the spectrum is sketched, not exact.")

# --------------------------------------------------------------------------
# 3. Direction-policy swap: TopComponent -> RandomTopK(3).
# --------------------------------------------------------------------------
# `RandomTopK(k)` picks a uniformly random direction among the top-k
# principal components. With ExactSVD the spectrum is deterministic, but
# the direction draw is RNG-driven, so different seeds produce different
# trees. This is the per-tree randomisation knob for a future PCA forest.
println("\n[3] Direction policy swap: TopComponent -> RandomTopK(3)")
randk_splitter = PCASplitter(
    ExactSVD(),
    RandomTopK(3),
    AnyOf(MaxLeafSize(32), IntrinsicDimRatio(0.1; n_floor = 256)),
    MedianSplit(),
)
tree_a = build_index(PCATreeIndex, data; splitter = randk_splitter,
                     rng = MersenneTwister(11))
tree_b = build_index(PCATreeIndex, data; splitter = randk_splitter,
                     rng = MersenneTwister(22))
ids_a = neighbor_ids(query(tree_a, data, query_vec, k))
ids_b = neighbor_ids(query(tree_b, data, query_vec, k))
println("    seed 11 recall: $(round(recall(query(tree_a, data, query_vec, k), truth) * 100, digits = 1))%")
println("    seed 22 recall: $(round(recall(query(tree_b, data, query_vec, k), truth) * 100, digits = 1))%")
println("    different seeds give different trees: $(ids_a != ids_b)")

# --------------------------------------------------------------------------
# 4. Stopping criterion swaps.
# --------------------------------------------------------------------------
# (a) Pure size cap — small leaves, deeper tree.
shallow_splitter = PCASplitter(
    ExactSVD(), TopComponent(), MaxLeafSize(16), MedianSplit(),
)
shallow_index = build_index(PCATreeIndex, data; splitter = shallow_splitter,
                            rng = MersenneTwister(1))
println("\n[4a] Stopping = MaxLeafSize(16)  (pure size cap, deeper tree)")
println("     n_nodes = $(length(shallow_index.nodes))  vs default $(length(default_index.nodes))")

# (b) Pure spectrum-flatness cap — stops only when the top PC explains
#     >= 1 - threshold of the variance, gated on n_node >= n_floor so
#     small-node sketches don't trigger spuriously. With our small
#     dataset (n=200) and n_floor=50, the criterion is active for the
#     root node but won't fire on subtrees below 50 points.
flat_splitter = PCASplitter(
    ExactSVD(), TopComponent(), IntrinsicDimRatio(0.05; n_floor = 50), MedianSplit(),
)
flat_index = build_index(PCATreeIndex, data; splitter = flat_splitter,
                         rng = MersenneTwister(1))
println("[4b] Stopping = IntrinsicDimRatio(0.05, n_floor=50)  (spectrum-flatness)")
println("     n_nodes = $(length(flat_index.nodes))")

# (c) AnyOf combinator — stop if EITHER rule fires. This is the default
#     style: a hard size cap as a safety net plus a spectrum-driven rule.
combo_splitter = PCASplitter(
    ExactSVD(), TopComponent(),
    AnyOf(MaxLeafSize(32), IntrinsicDimRatio(0.1; n_floor = 50)),
    MedianSplit(),
)
combo_index = build_index(PCATreeIndex, data; splitter = combo_splitter,
                          rng = MersenneTwister(1))
println("[4c] Stopping = AnyOf(MaxLeafSize(32), IntrinsicDimRatio(0.1, n_floor=50))")
println("     n_nodes = $(length(combo_index.nodes))   (AllOf is also available)")

# --------------------------------------------------------------------------
# 5. Split-value-policy swap: MedianSplit -> RandomBetweenQuantiles.
# --------------------------------------------------------------------------
# `RandomBetweenQuantiles(0.4, 0.6)` picks the threshold uniformly between
# the 40th and 60th projection percentiles. Trees stay roughly balanced
# but the threshold is randomised — another per-tree knob for forests.
println("\n[5] Split-value policy swap: MedianSplit -> RandomBetweenQuantiles(0.4, 0.6)")
rbq_splitter = PCASplitter(
    ExactSVD(), TopComponent(),
    AnyOf(MaxLeafSize(32), IntrinsicDimRatio(0.1; n_floor = 256)),
    RandomBetweenQuantiles(0.4, 0.6),
)
rbq_index = build_index(PCATreeIndex, data; splitter = rbq_splitter,
                        rng = MersenneTwister(7))
println("    recall@$k = $(round(recall(query(rbq_index, data, query_vec, k), truth) * 100, digits = 1))%")
println("    (MeanSplit is also available — useful when the projection is heavily skewed.)")

# --------------------------------------------------------------------------
# 6. Forest-ready recipe: `pca_forest_splitter()`.
# --------------------------------------------------------------------------
# This is the all-randomised configuration ready to drop into an
# eventual PCA forest wrapper:
#   - SubsampledSVD(2048, RandomizedSVD(rank=10))   sketch + subsample
#   - RandomTopK(3)                                 per-tree direction
#   - AnyOf(MaxLeafSize(32), IntrinsicDimRatio(...))
#   - MedianSplit
# Build N of these with distinct seeds and union the leaves, exactly as
# `RPTreeForestIndex` does for RP trees.
println("\n[6] Forest-ready recipe: pca_forest_splitter()")
forest_splitter = pca_forest_splitter(
    sample_cap = 128, rank = 8, n_iter = 2, top_k = 3,
    leaf_cap = 16, intrinsic_threshold = 0.1, n_floor = 50,
)
fr_a = build_index(PCATreeIndex, data; splitter = forest_splitter,
                   rng = MersenneTwister(101))
fr_b = build_index(PCATreeIndex, data; splitter = forest_splitter,
                   rng = MersenneTwister(202))
println("    seed 101 recall: $(round(recall(query(fr_a, data, query_vec, k), truth) * 100, digits = 1))%")
println("    seed 202 recall: $(round(recall(query(fr_b, data, query_vec, k), truth) * 100, digits = 1))%")
println("    Trees disagree at different seeds, so a future forest's union-of-leaves")
println("    candidate set will cover more of the true neighbourhood.")

# --------------------------------------------------------------------------
# 7. Extension story: writing your own stopping criterion.
# --------------------------------------------------------------------------
# Users can add their own policy without touching the trait machinery:
#
#     struct DepthCap <: AbstractStoppingCriterion
#         max_depth::Int
#     end
#     ManifoldANN.needs_spectrum(::DepthCap) = false
#     ManifoldANN.should_stop(c::DepthCap, n_node, _spectrum) = ...
#
# Same pattern for AbstractSpectrumEstimator (override estimate_spectrum),
# AbstractSplitDirectionPolicy (override pick_direction), and
# AbstractSplitValuePolicy (override pick_split_value). Compose with
# AnyOf/AllOf to mix new criteria with built-ins.

println("\n" * "=" ^ 70)
println("Walkthrough complete.")
println("=" ^ 70)
