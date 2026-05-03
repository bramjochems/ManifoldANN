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

# KDTree's pruning bound `|q[axis] - split_value| <= worst` is a valid lower
# bound on the full distance only for componentwise-monotone metrics — those
# where a per-axis difference can never exceed the full L_p-style aggregate.
# Cosine, Hamming, Jaccard, KL etc. violate this and would give silently wrong
# results. The build gate accepts the in-tree default helpers and the
# Distances.jl metrics that satisfy the property; everything else errors out
# at construction with a pointer to HNSW/LSH.
@inline _kdtree_safe_metric(::typeof(default_distance)) = true
@inline _kdtree_safe_metric(::typeof(default_squared_distance)) = true
@inline _kdtree_safe_metric(::Distances.Euclidean) = true
@inline _kdtree_safe_metric(::Distances.SqEuclidean) = true
@inline _kdtree_safe_metric(::Distances.Cityblock) = true
@inline _kdtree_safe_metric(::Distances.Chebyshev) = true
@inline _kdtree_safe_metric(::Distances.Minkowski) = true
@inline _kdtree_safe_metric(::Distances.WeightedEuclidean) = true
@inline _kdtree_safe_metric(::Distances.WeightedSqEuclidean) = true
@inline _kdtree_safe_metric(::Distances.WeightedCityblock) = true
@inline _kdtree_safe_metric(::Distances.WeightedMinkowski) = true
@inline _kdtree_safe_metric(_) = false
