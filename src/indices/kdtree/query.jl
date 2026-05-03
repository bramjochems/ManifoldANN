"""
    query(index::KDTreeIndex, data, q, k)

Exact kNN search using a balanced KD-tree. The pruning strategy assumes an
Euclidean-compatible metric (the index's `distance`); falls back to exploring
both subtrees when in doubt by comparing the query's coordinate offset to the
current worst neighbor distance.
"""
function query(
    index::KDTreeIndex{T,D},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer,
) where {T<:LinearAlgebra.BlasFloat,D}
    validate_index_dimensions(index, data, q)
    S = float(T)
    k <= 0 && return Neighbor{S}[]
    actual_k = min(Int(k), index.n_points)
    actual_k == 0 && return Neighbor{S}[]

    heap = BoundedMaxHeap{S}(actual_k)
    _search_kdtree!(index, index.root, data, q, heap)
    return to_sorted_vector(heap)
end

function _search_kdtree!(
    index::KDTreeIndex{T,D},
    node_id::Int,
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    heap::BoundedMaxHeap{S},
) where {T<:LinearAlgebra.BlasFloat,D,S<:AbstractFloat}
    node_id == 0 && return
    node = index.nodes[node_id]
    point_id = node.point_index
    dist = S(index.distance(@view(data[:, point_id]), q))
    push!(heap, point_id, dist)

    axis = node.axis
    split_value = node.split_value
    q_val = q[axis]
    go_left = q_val < split_value
    near_child = go_left ? node.left : node.right
    far_child = go_left ? node.right : node.left

    _search_kdtree!(index, near_child, data, q, heap)

    not_full = length(heap) < heap.capacity
    worst = not_full ? typemax(S) : peek_max(heap).dist
    axis_distance = abs(q_val - split_value)
    if axis_distance <= worst || not_full
        _search_kdtree!(index, far_child, data, q, heap)
    end
end

# Batch query: amortise the BoundedMaxHeap allocation by reusing one heap
# per task. Serial path uses a single heap; threaded path stashes one heap
# per scheduling task in TLS, keyed on (eltype, capacity).
function query(
    index::KDTreeIndex{T,D},
    data::AbstractMatrix{T},
    queries::AbstractMatrix{T},
    k::Integer,
) where {T<:LinearAlgebra.BlasFloat,D}
    size(queries, 1) == size(data, 1) ||
        throw(DimensionMismatch("Expected queries with $(size(data, 1)) rows, got $(size(queries, 1))"))
    n_queries = size(queries, 2)
    S = float(T)
    n_queries == 0 && return Vector{Vector{Neighbor{S}}}()
    results = Vector{Vector{Neighbor{S}}}(undef, n_queries)
    k <= 0 && (fill!(results, Neighbor{S}[]); return results)
    actual_k = min(Int(k), index.n_points)
    if actual_k == 0
        @inbounds for i in 1:n_queries
            results[i] = Neighbor{S}[]
        end
        return results
    end

    if Threads.nthreads() == 1 || n_queries < BATCH_THREAD_THRESHOLD
        heap = BoundedMaxHeap{S}(actual_k)
        @inbounds for i in 1:n_queries
            results[i] = _query_with_heap!(index, data, view(queries, :, i), actual_k, heap)
        end
    else
        Threads.@threads for i in 1:n_queries
            heap = _kdtree_task_heap(S, actual_k)
            @inbounds results[i] = _query_with_heap!(index, data, view(queries, :, i), actual_k, heap)
        end
    end
    return results
end

const _KDTREE_TASK_HEAP_KEY = :ManifoldANN_KDTree_task_heap

@inline function _kdtree_task_heap(::Type{S}, capacity::Int) where {S<:AbstractFloat}
    tls = task_local_storage()
    storage = get(tls, _KDTREE_TASK_HEAP_KEY, nothing)
    if storage isa BoundedMaxHeap{S} && storage.capacity == capacity
        empty!(storage.data)
        return storage::BoundedMaxHeap{S}
    end
    heap = BoundedMaxHeap{S}(capacity)
    tls[_KDTREE_TASK_HEAP_KEY] = heap
    return heap
end

@inline function _query_with_heap!(
    index::KDTreeIndex{T,D},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    actual_k::Int,
    heap::BoundedMaxHeap{S},
) where {T<:LinearAlgebra.BlasFloat,D,S<:AbstractFloat}
    validate_index_dimensions(index, data, q)
    empty!(heap.data)
    _search_kdtree!(index, index.root, data, q, heap)
    sorted = copy(heap.data)
    sort!(sorted, by = n -> (n.dist, n.id))
    return sorted
end