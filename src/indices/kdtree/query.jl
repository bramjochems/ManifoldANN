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

    buffer = NeighborBuffer(actual_k)
    _search_kdtree!(index, index.root, data, q, buffer, distance)
    return finalize_neighbors(buffer, S)
end

mutable struct NeighborBuffer
    ids::Vector{Int}
    dists::Vector{Float64}
    capacity::Int
end

NeighborBuffer(capacity::Int) = NeighborBuffer(Int[], Float64[], capacity)

@inline function buffer_length(buf::NeighborBuffer)
    return length(buf.ids)
end

function push_candidate!(buf::NeighborBuffer, id::Int, dist::Real)
    dist64 = Float64(dist)
    len = buffer_length(buf)
    if len < buf.capacity
        push!(buf.ids, id)
        push!(buf.dists, dist64)
        return
    end
    worst_dist, worst_idx = findmax(buf.dists)
    if dist64 < worst_dist
        buf.ids[worst_idx] = id
        buf.dists[worst_idx] = dist64
    end
end

function current_worst_distance(buf::NeighborBuffer)
    len = buffer_length(buf)
    len < buf.capacity && return Inf
    return maximum(buf.dists)
end

function finalize_neighbors(buf::NeighborBuffer, ::Type{S}) where {S<:AbstractFloat}
    len = buffer_length(buf)
    len == 0 && return Neighbor{S}[]
    order = sortperm(1:len; by = i -> (buf.dists[i], buf.ids[i]))
    results = Vector{Neighbor{S}}(undef, length(order))
    @inbounds for (pos, idx) in enumerate(order)
        results[pos] = Neighbor{S}(buf.ids[idx], S(buf.dists[idx]))
    end
    return results
end

function _search_kdtree!(
    index::KDTreeIndex{T},
    node_id::Int,
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    buffer::NeighborBuffer,
    distance::Function,
) where {T<:LinearAlgebra.BlasFloat}
    node_id == 0 && return
    node = index.nodes[node_id]
    point_id = node.point_index
    dist = distance(@view(data[:, point_id]), q)
    push_candidate!(buffer, point_id, dist)

    axis = node.axis
    split_value = node.split_value
    q_val = q[axis]
    go_left = q_val < split_value
    near_child = go_left ? node.left : node.right
    far_child = go_left ? node.right : node.left

    _search_kdtree!(index, near_child, data, q, buffer, distance)

    axis_distance = abs(q_val - split_value)
    if axis_distance <= current_worst_distance(buffer) || buffer_length(buffer) < buffer.capacity
        _search_kdtree!(index, far_child, data, q, buffer, distance)
    end
end
