"""
Shared neighbor storage data structures used across multiple index types.
"""

"""
    Neighbor{T}

Generic neighbor entry storing an identifier and distance.
Used as a building block for various neighbor storage structures.
"""
struct Neighbor{T<:AbstractFloat}
    id::Int
    dist::T
end

"""
    neighbor_ids(neighbors)

Extract the point identifiers stored in `neighbors`. The returned vector is a
copy so callers can mutate it without affecting the original neighbor list.
"""
@inline function neighbor_ids(neighbors::AbstractVector{<:Neighbor})
    ids = Vector{Int}(undef, length(neighbors))
    @inbounds for i in eachindex(neighbors)
        ids[i] = neighbors[i].id
    end
    return ids
end

"""
    neighbor_ids(neighbor_batches)

Extract neighbor identifiers for each query independently. Returns a vector of
`Vector{Int}` matching the structure of the input batches.
"""
function neighbor_ids(
    neighbor_batches::AbstractVector{<:AbstractVector{<:Neighbor}},
)
    results = Vector{Vector{Int}}(undef, length(neighbor_batches))
    @inbounds for i in eachindex(neighbor_batches)
        results[i] = neighbor_ids(neighbor_batches[i])
    end
    return results
end

"""
    BoundedMaxHeap{T}

Max-heap that maintains at most `capacity` elements, automatically evicting
the maximum (farthest) element when capacity is exceeded. Useful for maintaining
k-nearest neighbors where we want to efficiently track and remove the farthest neighbor.

Operations:
- `push!`: O(log n) - Add element, evict max if over capacity
- `peek_max`: O(1) - View the maximum element without removing
- `iterate`: O(n) - Iterate in no particular order
"""
struct BoundedMaxHeap{T<:AbstractFloat}
    data::Vector{Neighbor{T}}
    capacity::Int

    function BoundedMaxHeap{T}(capacity::Int) where {T<:AbstractFloat}
        capacity > 0 || throw(ArgumentError("capacity must be positive"))
        data = Neighbor{T}[]
        # Pre-size the underlying buffer to capacity so steady-state push! never
        # triggers _growend!/array_new_memory. Hot path in NN-Descent
        # _insert_neighbor!: profile showed Vector growth dominating self time.
        sizehint!(data, capacity)
        return new{T}(data, capacity)
    end
end

Base.length(heap::BoundedMaxHeap) = length(heap.data)
Base.isempty(heap::BoundedMaxHeap) = isempty(heap.data)
Base.iterate(heap::BoundedMaxHeap, state...) = iterate(heap.data, state...)

"""
    push!(heap::BoundedMaxHeap, id::Int, dist)

Insert a neighbor into the heap. If the heap is at capacity and the new neighbor
is closer than the current maximum, the maximum is removed and the new neighbor is added.
Returns true if the element was added, false if it was rejected.
"""
function Base.push!(heap::BoundedMaxHeap{T}, id::Int, dist::T) where {T}
    neighbor = Neighbor{T}(id, dist)

    # If not at capacity, just add
    if length(heap) < heap.capacity
        push!(heap.data, neighbor)
        _heap_sift_up!(heap.data, length(heap.data))
        return true
    end

    # If at capacity, only add if better than current max
    if dist < heap.data[1].dist
        heap.data[1] = neighbor
        _heap_sift_down!(heap.data, 1)
        return true
    end

    return false
end

"""
    unsafe_push!(heap::BoundedMaxHeap, id::Int, dist)

Insert a neighbor into the heap WITHOUT capacity checks. This allows the heap
to exceed its capacity limit. Used during symmetrization where we need to add
reverse edges regardless of capacity.

WARNING: This breaks the "bounded" invariant. Only use when you plan to prune later.
"""
function unsafe_push!(heap::BoundedMaxHeap{T}, id::Int, dist::T) where {T}
    neighbor = Neighbor{T}(id, dist)
    push!(heap.data, neighbor)
    _heap_sift_up!(heap.data, length(heap.data))
    return nothing
end

"""
    peek_max(heap::BoundedMaxHeap)

Return the maximum (farthest) element without removing it.
Throws if heap is empty.
"""
function peek_max(heap::BoundedMaxHeap)
    isempty(heap) && throw(ArgumentError("Cannot peek at empty BoundedMaxHeap"))
    return heap.data[1]
end

"""
    to_sorted_vector(heap::BoundedMaxHeap)

Extract all neighbors sorted by distance (closest first).
"""
function to_sorted_vector(heap::BoundedMaxHeap{T}) where {T}
    sorted = copy(heap.data)
    sort!(sorted, by = n -> n.dist)
    return sorted
end

"""
    contains_id(heap::BoundedMaxHeap, id::Int) -> Bool

Check if a neighbor with the given id exists in the heap. O(n) operation.
"""
function contains_id(heap::BoundedMaxHeap, id::Int)
    for neighbor in heap.data
        neighbor.id == id && return true
    end
    return false
end

# Max-heap internal operations
@inline function _heap_sift_up!(data::Vector{Neighbor{T}}, idx::Int) where {T}
    while idx > 1
        parent = idx >>> 1
        # Max-heap: parent should be >= child
        if data[idx].dist > data[parent].dist
            data[idx], data[parent] = data[parent], data[idx]
            idx = parent
        else
            break
        end
    end
    return nothing
end

@inline function _heap_sift_down!(data::Vector{Neighbor{T}}, idx::Int) where {T}
    len = length(data)
    while true
        left = idx << 1
        right = left + 1
        largest = idx

        # Max-heap: find largest among node and children
        if left <= len && data[left].dist > data[largest].dist
            largest = left
        end
        if right <= len && data[right].dist > data[largest].dist
            largest = right
        end

        largest == idx && break
        data[idx], data[largest] = data[largest], data[idx]
        idx = largest
    end
    return nothing
end
