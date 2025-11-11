using LinearAlgebra
using Random

const NeighborList = Vector{Int}

"""
    NeighborCandidate

Helper struct carrying a node id plus its distance to the current query. Used
throughout traversal and connection logic to keep code type-stable.
"""
struct NeighborCandidate
    id::Int
    dist::Float64
end

"""
    HNSWLayer

Adjacency representation for a single HNSW layer. Each node owns a mutable
vector of neighbor ids.
"""
const HNSWLayer = Vector{NeighborList}

"""
    HNSWIndex

Hierarchical Navigable Small World graph-based index with pluggable layer
planners, neighbor-selection policies, and traversal strategies. Stores only
the structural graph metadata so callers remain responsible for supplying the
dataset at query time.
"""
mutable struct HNSWIndex{T<:LinearAlgebra.BlasFloat,LP,NP,TP} <: AbstractANNIndex
    layers::Vector{HNSWLayer}
    entry_point::Int
    max_layer::Int
    dimension::Int
    n_points::Int
    M::Int
    ef_construction::Int
    planner::LP
    neighbor_policy::NP
    traversal_policy::TP
end

configured_k(::HNSWIndex) = nothing
supports_mutation(::HNSWIndex) = true
