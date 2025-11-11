using LinearAlgebra
using Random

const NeighborList = Vector{Int}

"""
    NeighborCandidate{T}

Helper struct carrying a node id plus its distance to the current query. The
distance type `T` remains parametric so we avoid widening (e.g., Float32 → Float64)
when the active distance metric already matches the data precision.
"""
struct NeighborCandidate{T<:AbstractFloat}
    id::Int
    dist::T
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

# Type Parameters
- `T`: Element type (e.g., Float32, Float64)
- `LP`: Layer planner type
- `NP`: Neighbor policy type
- `TP`: Traversal policy type
- `D`: Distance function type (must be thread-safe)
"""
mutable struct HNSWIndex{T<:LinearAlgebra.BlasFloat,LP,NP,TP,D} <: AbstractANNIndex
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
    distance::D
end

configured_k(::HNSWIndex) = nothing
supports_mutation(::HNSWIndex) = true
