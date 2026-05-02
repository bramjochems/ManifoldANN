"""
    query(index::KDTreeIndex, data, q, k; distance=default_distance)

Exact kNN search using a balanced KD-tree. The pruning strategy assumes an
Euclidean-compatible metric (the default `default_distance`), but falls back
to exploring both subtrees when in doubt by comparing the query's coordinate
offset to the current worst neighbor distance.
"""
function query(
    index::KDTreeIndex{T},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer;
    distance::Function = default_distance,
) where {T<:LinearAlgebra.BlasFloat}
    validate_index_dimensions(index, data, q)
    S = float(T)
    k <= 0 && return Neighbor{S}[]
    actual_k = min(Int(k), index.n_points)
    actual_k == 0 && return Neighbor{S}[]

    heap = BoundedMaxHeap{S}(actual_k)
    _search_kdtree!(index, index.root, data, q, heap, distance)
    results = to_sorted_vector(heap)
    sort!(results, by = n -> (n.dist, n.id))
    return results
end

function _search_kdtree!(
    index::KDTreeIndex{T},
    node_id::Int,
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    heap::BoundedMaxHeap{S},
    distance::Function,
) where {T<:LinearAlgebra.BlasFloat,S<:AbstractFloat}
    node_id == 0 && return
    node = index.nodes[node_id]
    point_id = node.point_index
    dist = S(distance(@view(data[:, point_id]), q))
    push!(heap, point_id, dist)

    axis = node.axis
    split_value = node.split_value
    q_val = q[axis]
    go_left = q_val < split_value
    near_child = go_left ? node.left : node.right
    far_child = go_left ? node.right : node.left

    _search_kdtree!(index, near_child, data, q, heap, distance)

    not_full = length(heap) < heap.capacity
    worst = not_full ? typemax(S) : peek_max(heap).dist
    axis_distance = abs(q_val - split_value)
    if axis_distance <= worst || not_full
        _search_kdtree!(index, far_child, data, q, heap, distance)
    end
end
