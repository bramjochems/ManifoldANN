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
    pending::NeighborMinHeap{T}           # min-first queue
    best::Vector{NeighborCandidate{T}}    # sorted ascending by distance
end

GreedyTraversalPolicy(; ef_search::Int = 64) = GreedyTraversalPolicy(ef_search)

function initialize_state(::GreedyTraversalPolicy, entry::NeighborCandidate{T}) where {T}
    return TraversalState{T}(NeighborMinHeap(entry), [entry])
end

function should_continue(::GreedyTraversalPolicy, state::TraversalState)
    return !isempty(state.pending)
end

function pop_pending!(::GreedyTraversalPolicy, state::TraversalState)
    return popfirst!(state.pending)
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
    push!(state.pending, candidate)
    _insert_sorted!(state.best, candidate)
    if length(state.best) > policy.ef_search
        pop!(state.best) # drop worst
    end
end

default_ef(policy::GreedyTraversalPolicy) = policy.ef_search
with_ef(::GreedyTraversalPolicy, ef::Int) = GreedyTraversalPolicy(ef)

@inline function _insert_sorted!(
    vec::Vector{NeighborCandidate{T}},
    candidate::NeighborCandidate{T},
) where {T}
    idx = searchsortedlast(vec, candidate; by = c -> c.dist)
    insert!(vec, idx + 1, candidate)
    return vec
end
