using LinearAlgebra

"""
    KDTreeNode

Axis-aligned KD-tree node storing the splitting dimension, split value,
child references (by index into the node vector), and the dataset point
anchored at this node.
"""
struct KDTreeNode{T<:LinearAlgebra.BlasFloat}
    axis::Int
    point_index::Int
    split_value::T
    left::Int
    right::Int
end

"""
    KDTreeIndex

Balanced KD-tree built over a static dataset. Stores only structural metadata
(split axes, thresholds, and point identifiers) so callers can supply any
matrix with matching coordinates at query time.
"""
mutable struct KDTreeIndex{T<:LinearAlgebra.BlasFloat,D} <: AbstractANNIndex
    nodes::Vector{KDTreeNode{T}}
    dimension::Int
    n_points::Int
    root::Int
    distance::D
end

index_distance(index::KDTreeIndex) = index.distance
configured_k(::KDTreeIndex) = nothing
supports_mutation(::KDTreeIndex) = false
