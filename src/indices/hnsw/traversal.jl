abstract type AbstractTraversalPolicy end

default_ef(::AbstractTraversalPolicy) =
    error("Traversal policy must implement `default_ef`")
with_ef(::AbstractTraversalPolicy, ::Int) =
    error("Traversal policy must implement `with_ef`")

"""
    GreedyTraversalPolicy(ef_search)

Classic HNSW traversal policy that uses greedy descent on upper layers and a
best-first expansion (controlled by `ef_search`) on the target layer.
"""
struct GreedyTraversalPolicy <: AbstractTraversalPolicy
    ef_search::Int
end

struct TraversalState{T}
    pending::Vector{NeighborCandidate{T}} # min-first queue
    best::Vector{NeighborCandidate{T}}    # sorted ascending by distance
end

GreedyTraversalPolicy(; ef_search::Int = 64) = GreedyTraversalPolicy(ef_search)

function initialize_state(::GreedyTraversalPolicy, entry::NeighborCandidate{T}) where {T}
    return TraversalState{T}([entry], [entry])
end

function should_continue(::GreedyTraversalPolicy, state::TraversalState)
    return !isempty(state.pending)
end

function pop_pending!(::GreedyTraversalPolicy, state::TraversalState)
    current = popfirst!(state.pending)
    return current
end

function worst_distance(policy::GreedyTraversalPolicy, state::TraversalState)
    len = length(state.best)
    len < policy.ef_search && return Inf
    return state.best[end].dist
end

function maybe_push_candidate!(
    policy::GreedyTraversalPolicy,
    state::TraversalState{T},
    candidate::NeighborCandidate{T},
) where {T}
    # pending queue sorted ascending by distance
    push!(state.pending, candidate)
    sort!(state.pending, by = c -> c.dist)

    push!(state.best, candidate)
    sort!(state.best, by = c -> c.dist)
    if length(state.best) > policy.ef_search
        pop!(state.best) # drop worst
    end
end

default_ef(policy::GreedyTraversalPolicy) = policy.ef_search
with_ef(::GreedyTraversalPolicy, ef::Int) = GreedyTraversalPolicy(ef)
