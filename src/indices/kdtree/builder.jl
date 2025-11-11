"""
    build_index(KDTreeIndex, data; axis_selector = :variance)

Construct a balanced KD-tree over `data`. Only metadata required for search
is retained inside the index, so callers are still responsible for supplying
the point matrix when querying.

`axis_selector` controls how split dimensions are chosen:
- `:variance` (default): pick the axis with the largest spread per subtree
- `:cyclic`: cycle through axes based on recursion depth
"""
function build_index(
    ::Type{KDTreeIndex},
    data::AbstractMatrix{T};
    axis_selector::Symbol = :variance,
) where {T<:LinearAlgebra.BlasFloat}
    d, n = size(data)
    d > 0 || throw(ArgumentError("Dataset must have at least one dimension"))
    n > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    axis_selector in (:variance, :cyclic) ||
        throw(ArgumentError("axis_selector must be :variance or :cyclic"))

    indices = collect(1:n)
    nodes = Vector{KDTreeNode{T}}()
    sizehint!(nodes, n)
    root = _build_kdtree!(nodes, data, indices, axis_selector, 1)
    return KDTreeIndex{T}(nodes, d, n, root)
end

function _build_kdtree!(
    nodes::Vector{KDTreeNode{T}},
    data::AbstractMatrix{T},
    indices::Vector{Int},
    axis_selector::Symbol,
    depth::Int,
) where {T<:LinearAlgebra.BlasFloat}
    isempty(indices) && return 0

    axis = _pick_axis(data, indices, axis_selector, depth)
    sort!(indices, by = i -> data[axis, i])
    median_pos = (length(indices) + 1) >>> 1
    point_index = indices[median_pos]
    split_value = data[axis, point_index]

    node_placeholder = KDTreeNode{T}(axis, point_index, split_value, 0, 0)
    push!(nodes, node_placeholder)
    node_id = length(nodes)

    left_indices = median_pos > 1 ? Vector{Int}(indices[1:median_pos-1]) : Int[]
    right_indices =
        median_pos < length(indices) ? Vector{Int}(indices[median_pos+1:end]) : Int[]

    left_child = _build_kdtree!(nodes, data, left_indices, axis_selector, depth + 1)
    right_child = _build_kdtree!(nodes, data, right_indices, axis_selector, depth + 1)

    nodes[node_id] = KDTreeNode{T}(axis, point_index, split_value, left_child, right_child)
    return node_id
end

function _pick_axis(
    data::AbstractMatrix{T},
    indices::Vector{Int},
    axis_selector::Symbol,
    depth::Int,
) where {T<:LinearAlgebra.BlasFloat}
    if axis_selector === :cyclic
        d = size(data, 1)
        return ((depth - 1) % d) + 1
    end
    return _axis_with_max_spread(data, indices)
end

function _axis_with_max_spread(data::AbstractMatrix, indices::Vector{Int})
    d = size(data, 1)
    best_axis = 1
    best_span = -Inf
    for axis in 1:d
        min_val = Inf
        max_val = -Inf
        @inbounds for idx in indices
            value = data[axis, idx]
            if value < min_val
                min_val = value
            end
            if value > max_val
                max_val = value
            end
        end
        span = max_val - min_val
        if span > best_span
            best_span = span
            best_axis = axis
        end
    end
    return best_axis
end
