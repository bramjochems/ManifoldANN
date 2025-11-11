"""
    NeighborMinHeap{T}

Binary min-heap specialized for `NeighborCandidate{T}`. Provides the same
push/pop-first semantics the traversal policy expects, but avoids repeatedly
sorting the pending queue.
"""
struct NeighborMinHeap{T<:AbstractFloat}
    data::Vector{NeighborCandidate{T}}
end

NeighborMinHeap{T}() where {T<:AbstractFloat} =
    NeighborMinHeap{T}(NeighborCandidate{T}[])

function NeighborMinHeap(entry::NeighborCandidate{T}) where {T<:AbstractFloat}
    heap = NeighborMinHeap{T}()
    push!(heap, entry)
    return heap
end

Base.length(heap::NeighborMinHeap) = length(heap.data)
Base.isempty(heap::NeighborMinHeap) = isempty(heap.data)

function Base.push!(heap::NeighborMinHeap{T}, candidate::NeighborCandidate{T}) where {T}
    push!(heap.data, candidate)
    _heap_sift_up!(heap.data, length(heap.data))
    return heap
end

function Base.popfirst!(heap::NeighborMinHeap{T}) where {T}
    isempty(heap) && throw(ArgumentError("Cannot pop from an empty NeighborMinHeap"))
    top = heap.data[1]
    last = pop!(heap.data)
    if !isempty(heap)
        heap.data[1] = last
        _heap_sift_down!(heap.data, 1)
    end
    return top
end

@inline function _heap_sift_up!(data::Vector{NeighborCandidate{T}}, idx::Int) where {T}
    while idx > 1
        parent = idx >>> 1
        if data[idx].dist < data[parent].dist
            data[idx], data[parent] = data[parent], data[idx]
            idx = parent
        else
            break
        end
    end
    return nothing
end

@inline function _heap_sift_down!(data::Vector{NeighborCandidate{T}}, idx::Int) where {T}
    len = length(data)
    while true
        left = idx << 1
        right = left + 1
        smallest = idx

        if left <= len && data[left].dist < data[smallest].dist
            smallest = left
        end
        if right <= len && data[right].dist < data[smallest].dist
            smallest = right
        end

        smallest == idx && break
        data[idx], data[smallest] = data[smallest], data[idx]
        idx = smallest
    end
    return nothing
end
