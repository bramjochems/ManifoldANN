abstract type AbstractNeighborPolicy end

function max_degree(::AbstractNeighborPolicy)
    error("Neighbor policy must implement `max_degree`")
end

"""
    HeuristicNeighborPolicy

Simplified neighbor selection strategy that keeps the closest `M` candidates.
`M_max` lets callers enforce a hard upper bound per node (default = `M`).
"""
struct HeuristicNeighborPolicy <: AbstractNeighborPolicy
    M::Int
    M_max::Int
end

HeuristicNeighborPolicy(M::Int) = HeuristicNeighborPolicy(M, M)

function select_neighbors(policy::HeuristicNeighborPolicy, candidates::Vector{NeighborCandidate})
    isempty(candidates) && return NeighborCandidate[]
    limit = min(policy.M, length(candidates))
    # Use partialsort! to only sort the top M elements (O(n) vs O(n log n))
    partialsort!(candidates, 1:limit, by = c -> c.dist)
    return copy(candidates[1:limit])
end

function max_degree(policy::HeuristicNeighborPolicy)
    return policy.M_max
end
