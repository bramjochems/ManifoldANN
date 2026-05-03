"""
    build_index(KDTreeIndex, data; axis_selector = :variance, leafsize = KDTREE_DEFAULT_LEAFSIZE)

Construct a balanced KD-tree over `data`. Internal nodes are pure routers
(split axis + threshold); points live in leaf buckets of size up to
`leafsize`. Only metadata required for search is retained inside the
index, so callers are still responsible for supplying the point matrix
when querying.

`axis_selector` controls how split dimensions are chosen:
- `:variance` (default): pick the axis with the largest spread per subtree
- `:cyclic`: cycle through axes based on recursion depth
"""
function build_index(
    ::Type{KDTreeIndex},
    data::AbstractMatrix{T};
    axis_selector::Symbol = :variance,
    leafsize::Int = KDTREE_DEFAULT_LEAFSIZE,
    distance::D = default_distance,
) where {T<:LinearAlgebra.BlasFloat,D}
    d, n = size(data)
    d > 0 || throw(ArgumentError("Dataset must have at least one dimension"))
    n > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    leafsize >= 1 || throw(ArgumentError("leafsize must be >= 1"))
    axis_selector in (:variance, :cyclic) ||
        throw(ArgumentError("axis_selector must be :variance or :cyclic"))
    _kdtree_safe_metric(distance) || throw(ArgumentError(
        "KDTreeIndex's `axis_distance ≤ worst` pruning bound is only correct " *
        "for componentwise-monotone metrics (Euclidean, SqEuclidean, Cityblock, " *
        "Chebyshev, Minkowski). Got distance=$distance — for cosine, hamming, " *
        "or other non-monotone metrics use HNSWIndex or LSHIndex instead."))

    indices = collect(1:n)
    nodes = Vector{KDTreeNode{T}}()
    sizehint!(nodes, max(1, 2 * (n ÷ max(leafsize, 1)) - 1))
    root = _build_kdtree!(nodes, data, indices, 1, n, axis_selector, leafsize, 1)
    return KDTreeIndex{T,D}(nodes, indices, d, n, root, leafsize, distance)
end

function _build_kdtree!(
    nodes::Vector{KDTreeNode{T}},
    data::AbstractMatrix{T},
    indices::Vector{Int},
    lo::Int,
    hi::Int,
    axis_selector::Symbol,
    leafsize::Int,
    depth::Int,
) where {T<:LinearAlgebra.BlasFloat}
    lo > hi && return 0

    # Leaf: range fits in a bucket, no further split.
    if hi - lo + 1 <= leafsize
        push!(nodes, KDTreeNode{T}(0, zero(T), lo, hi))
        return length(nodes)
    end

    axis = _pick_axis(data, indices, lo, hi, axis_selector, depth)
    len = hi - lo + 1
    median_pos = lo + ((len + 1) >>> 1) - 1
    _quickselect_axis!(indices, lo, hi, median_pos, data, axis)
    split_value = data[axis, indices[median_pos]]

    # Internal node: pure router. The median-positioned point goes into the
    # right subtree's leaf via `lo = median_pos`.
    push!(nodes, KDTreeNode{T}(axis, split_value, 0, 0))
    node_id = length(nodes)

    left_child  = _build_kdtree!(nodes, data, indices, lo, median_pos - 1, axis_selector, leafsize, depth + 1)
    right_child = _build_kdtree!(nodes, data, indices, median_pos, hi, axis_selector, leafsize, depth + 1)

    nodes[node_id] = KDTreeNode{T}(axis, split_value, left_child, right_child)
    return node_id
end

function _pick_axis(
    data::AbstractMatrix{T},
    indices::Vector{Int},
    lo::Int,
    hi::Int,
    axis_selector::Symbol,
    depth::Int,
) where {T<:LinearAlgebra.BlasFloat}
    if axis_selector === :cyclic
        d = size(data, 1)
        return ((depth - 1) % d) + 1
    end
    return _axis_with_max_spread(data, indices, lo, hi)
end

# Hoare-partition quickselect on `indices[lo:hi]` keyed by `data[axis, *]`.
# Median-of-three pivot avoids O(n^2) on sorted/adversarial inputs.
@inline function _quickselect_axis!(
    indices::Vector{Int},
    lo::Int,
    hi::Int,
    target::Int,
    data::AbstractMatrix,
    axis::Int,
)
    @inbounds while lo < hi
        if hi - lo < 16
            for i in (lo + 1):hi
                idx_i = indices[i]
                key = data[axis, idx_i]
                j = i - 1
                while j >= lo && data[axis, indices[j]] > key
                    indices[j + 1] = indices[j]
                    j -= 1
                end
                indices[j + 1] = idx_i
            end
            return
        end
        mid = (lo + hi) >>> 1
        a = data[axis, indices[lo]]
        b = data[axis, indices[mid]]
        c = data[axis, indices[hi]]
        if a > b
            indices[lo], indices[mid] = indices[mid], indices[lo]
            a, b = b, a
        end
        if a > c
            indices[lo], indices[hi] = indices[hi], indices[lo]
            a, c = c, a
        end
        if b > c
            indices[mid], indices[hi] = indices[hi], indices[mid]
            b, c = c, b
        end
        indices[mid], indices[hi - 1] = indices[hi - 1], indices[mid]
        pivot_idx = indices[hi - 1]
        pivot_val = data[axis, pivot_idx]
        i = lo
        j = hi - 1
        while true
            i += 1
            while data[axis, indices[i]] < pivot_val
                i += 1
            end
            j -= 1
            while data[axis, indices[j]] > pivot_val
                j -= 1
            end
            if i >= j
                break
            end
            indices[i], indices[j] = indices[j], indices[i]
        end
        indices[i], indices[hi - 1] = indices[hi - 1], indices[i]
        if target == i
            return
        elseif target < i
            hi = i - 1
        else
            lo = i + 1
        end
    end
    return
end

function _axis_with_max_spread(data::AbstractMatrix, indices::Vector{Int}, lo::Int, hi::Int)
    d = size(data, 1)
    best_axis = 1
    best_span = -Inf
    for axis in 1:d
        min_val = Inf
        max_val = -Inf
        @inbounds for k in lo:hi
            idx = indices[k]
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
