"""
    RPTreeIndex{T,D}

Single random-projection tree wrapped as an `AbstractANNIndex`. Routes a
query to its leaf bucket and brute-force scans the bucket. Recall is
inherently lower than the forest variant (`RPTreeForestIndex`); this
type is the per-tree primitive the forest composes.
"""
mutable struct RPTreeIndex{T<:AbstractFloat,D} <: AbstractANNIndex
    tree::RPTree{T}
    dimension::Int
    n_points::Int
    leaf_cap::Int
    distance::D
end

index_distance(index::RPTreeIndex) = index.distance
configured_k(::RPTreeIndex) = nothing
supports_mutation(::RPTreeIndex) = false

const RPTREE_INDEX_DEFAULT_LEAF_CAP = 32
