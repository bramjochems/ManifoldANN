using Base: BitSet

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

function select_neighbors(
    policy::HeuristicNeighborPolicy,
    candidates::Vector{NeighborCandidate{T}},
    ::AbstractMatrix,
    _distance;
    limit::Union{Nothing,Int} = nothing,
) where {T}
    isempty(candidates) && return NeighborCandidate{T}[]
    cap = limit === nothing ? policy.M : limit
    cap = min(cap, length(candidates))
    # Use partialsort! to only sort the top M elements (O(n) vs O(n log n))
    partialsort!(candidates, 1:cap, by = c -> c.dist)
    # Return slice directly - candidates is always a temporary vector at call sites
    return candidates[1:cap]
end

function max_degree(policy::HeuristicNeighborPolicy)
    return policy.M_max
end

"""
    DiversifiedNeighborPolicy

Neighbor selection strategy mirroring Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC)
from the HNSW paper. It favours diverse long-range connections by discarding
candidates that are closer to an already-picked neighbor than to the query,
which improves recall. When the diversified pass selects fewer than `cap`
neighbors, the remaining slots are filled from the rejected candidates by
distance order (the `keepPrunedConnections` fallback in the paper).
"""
struct DiversifiedNeighborPolicy <: AbstractNeighborPolicy
    M::Int
    M_max::Int
end

DiversifiedNeighborPolicy(M::Int) = DiversifiedNeighborPolicy(M, M)

function select_neighbors(
    policy::DiversifiedNeighborPolicy,
    candidates::Vector{NeighborCandidate{T}},
    data::AbstractMatrix,
    distance_fn;
    limit::Union{Nothing,Int} = nothing,
) where {T}
    isempty(candidates) && return NeighborCandidate{T}[]

    cap = limit === nothing ? policy.M : limit

    # Short-circuit: if we have few enough candidates, no diversification needed
    if length(candidates) <= cap
        return candidates
    end

    sorted = sort(candidates; alg = Base.Sort.MergeSort, by = c -> c.dist)
    selected = Vector{NeighborCandidate{T}}()
    selected_ids = BitSet()
    cap = min(cap, length(sorted))

    for cand in sorted
        dominated = false
        for chosen in selected
            dist_bc = distance_fn(@view(data[:, cand.id]), @view(data[:, chosen.id]))
            if dist_bc < cand.dist
                dominated = true
                break
            end
        end

        if !dominated
            push!(selected, cand)
            push!(selected_ids, cand.id)
        end

        length(selected) == cap && break
    end

    if length(selected) < cap
        for cand in sorted
            cand.id in selected_ids && continue
            push!(selected, cand)
            push!(selected_ids, cand.id)
            length(selected) == cap && break
        end
    end

    return selected
end

function max_degree(policy::DiversifiedNeighborPolicy)
    return policy.M_max
end
