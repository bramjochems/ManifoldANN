using LinearAlgebra
using Distances

"""
    KDTreeNode

KD-tree node. Two roles, distinguished by `axis`:

- **Internal node** (`axis >= 1`): a pure router. Stores splitting axis,
  split value, and child node indices. Does NOT store a data point — there
  is no per-internal-node distance computation. All data points live in
  leaves.
- **Leaf node** (`axis == 0`): stores a range `[bucket_lo, bucket_hi]` (both
  inclusive) into `KDTreeIndex.indices`. At query time the leaf is scanned
  linearly: the bucket members are `indices[bucket_lo:bucket_hi]`.

`split_value` is unused for leaves; `bucket_lo` is stored in `left`,
`bucket_hi` in `right`.
"""
struct KDTreeNode{T<:LinearAlgebra.BlasFloat}
    axis::Int           # >=1 internal (split axis); 0 leaf
    split_value::T      # internal: split threshold; leaf: unused
    left::Int           # internal: left child id; leaf: bucket_lo
    right::Int          # internal: right child id; leaf: bucket_hi
end

@inline is_leaf(node::KDTreeNode) = node.axis == 0

"""
    KDTreeIndex

Balanced KD-tree built over a static dataset. Internal nodes are pure
routers (no anchor point); points live in leaf buckets. The `indices`
permutation gives bucket membership: leaf node `n` owns the points
`indices[n.left:n.right]`.
"""
mutable struct KDTreeIndex{T<:LinearAlgebra.BlasFloat,D} <: AbstractANNIndex
    nodes::Vector{KDTreeNode{T}}
    indices::Vector{Int}    # permutation of 1:n_points; leaf buckets are ranges
    dimension::Int
    n_points::Int
    root::Int
    leafsize::Int
    distance::D
end

# Default leaf size — chosen as a sensible non-tunable constant. ~16 is the
# usual sweet spot in KD-tree literature; larger reduces internal-node
# overhead but starts losing pruning effectiveness, smaller approaches the
# old per-point recursion.
const KDTREE_DEFAULT_LEAFSIZE = 16

index_distance(index::KDTreeIndex) = index.distance
configured_k(::KDTreeIndex) = nothing
supports_mutation(::KDTreeIndex) = false

# KDTree's prune bound is built from per-axis contributions of an
# axis-aligned cell containing the candidate points. A metric is safe iff
# its full distance is bounded below by an additive (or max-style) reduce
# of those per-axis contributions — the Friedman-Bentley-Finkel 1977
# safety class. Cosine, Hamming, Jaccard, KL etc. violate this and would
# give silently wrong results.
#
# `SqEuclidean` is on the safe list: the query path uses an incremental
# rolling cell-distance bound (FBF77) computed in the metric's prune units
# (squared for L2/SqL2, linear for L1/Minkowski), so the prune compare is
# unit-consistent. `WeightedEuclidean`, `WeightedCityblock`, and
# `WeightedMinkowski` are also on the rolling-bound path: per-axis
# contribution is `w[axis] * excess^p`, so weights < 1 are handled
# correctly (the legacy `axis_distance <= worst` prune over-prunes when
# weights < 1 and is no longer used for them). See
# `_kdtree_use_rolling_bound` in `query.jl` for the metrics that take the
# rolling-bound path. Chebyshev currently still falls back to the legacy
# linear-axis prune.
@inline _kdtree_safe_metric(::Distances.Euclidean) = true
@inline _kdtree_safe_metric(::Distances.SqEuclidean) = true
@inline _kdtree_safe_metric(::Distances.Cityblock) = true
@inline _kdtree_safe_metric(::Distances.Chebyshev) = true
@inline _kdtree_safe_metric(::Distances.Minkowski) = true
@inline _kdtree_safe_metric(::Distances.WeightedEuclidean) = true
@inline _kdtree_safe_metric(::Distances.WeightedCityblock) = true
@inline _kdtree_safe_metric(::Distances.WeightedMinkowski) = true
@inline _kdtree_safe_metric(_) = false
