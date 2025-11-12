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

"""
    BestCandidatesHeap{T}

Max-heap that maintains the best (closest) candidates, automatically evicting
the worst (farthest) when capacity is exceeded. More efficient than maintaining
a sorted vector when doing many insertions.
"""
struct BestCandidatesHeap{T<:AbstractFloat}
    data::Vector{NeighborCandidate{T}}
    capacity::Int
end

Base.length(heap::BestCandidatesHeap) = length(heap.data)
Base.isempty(heap::BestCandidatesHeap) = isempty(heap.data)

function Base.push!(heap::BestCandidatesHeap{T}, candidate::NeighborCandidate{T}) where {T}
    # If not at capacity, just add
    if length(heap.data) < heap.capacity
        push!(heap.data, candidate)
        _heap_sift_up_candidates!(heap.data, length(heap.data))
        return true
    end

    # If at capacity, only add if better than current max (farthest)
    if candidate.dist < heap.data[1].dist
        heap.data[1] = candidate
        _heap_sift_down_candidates!(heap.data, 1)
        return true
    end

    return false
end

# Get worst (farthest) distance - max at heap root
worst_distance(heap::BestCandidatesHeap) = isempty(heap.data) ? Inf : heap.data[1].dist

# Heap operations for NeighborCandidate (max-heap by distance)
function _heap_sift_up_candidates!(data::Vector{NeighborCandidate{T}}, idx::Int) where {T}
    while idx > 1
        parent = idx ÷ 2
        if data[idx].dist > data[parent].dist
            data[idx], data[parent] = data[parent], data[idx]
            idx = parent
        else
            break
        end
    end
end

function _heap_sift_down_candidates!(data::Vector{NeighborCandidate{T}}, idx::Int) where {T}
    n = length(data)
    while true
        largest = idx
        left = 2 * idx
        right = 2 * idx + 1

        if left <= n && data[left].dist > data[largest].dist
            largest = left
        end
        if right <= n && data[right].dist > data[largest].dist
            largest = right
        end

        if largest != idx
            data[idx], data[largest] = data[largest], data[idx]
            idx = largest
        else
            break
        end
    end
end

struct TraversalState{T}
    pending::NeighborMinHeap{T}                    # min-first queue
    best::BestCandidatesHeap{T}                    # max-heap of closest candidates
end

GreedyTraversalPolicy(; ef_search::Int = 64) = GreedyTraversalPolicy(ef_search)

function initialize_state(policy::GreedyTraversalPolicy, entry::NeighborCandidate{T}) where {T}
    best_heap = BestCandidatesHeap{T}(NeighborCandidate{T}[], policy.ef_search)
    push!(best_heap, entry)
    return TraversalState{T}(NeighborMinHeap(entry), best_heap)
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
    return worst_distance(state.best)
end

function maybe_push_candidate!(
    ::GreedyTraversalPolicy,
    state::TraversalState{T},
    candidate::NeighborCandidate{T},
) where {T}
    # Only add to pending queue if it was accepted into the best heap
    # This keeps the search focused on promising regions
    if push!(state.best, candidate)
        push!(state.pending, candidate)
    end
end

default_ef(policy::GreedyTraversalPolicy) = policy.ef_search
with_ef(::GreedyTraversalPolicy, ef::Int) = GreedyTraversalPolicy(ef)
