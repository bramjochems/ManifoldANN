"""
    query(index::KDTreeIndex, data, q, k)

Exact kNN search using a balanced KD-tree with leaf buckets. Internal
nodes prune via the **Friedman-Bentley-Finkel (1977) rolling cell
bound**: a squared-Euclidean (or, for non-L2 metrics, metric-native)
lower bound on the distance from `q` to any point inside the current
cell, updated incrementally — only the split axis's contribution
changes when descending — so the prune compare is in matched units.
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
    _kdtree_descent!(index, data, q, heap)
    return to_sorted_vector(heap)
end

# --- Per-metric rolling-bound primitives ---------------------------------
#
# A safe metric must satisfy: for any axis-aligned cell C and point p in C,
# `dist(q, p) >= cell_lb(q, C)` where `cell_lb` is built from per-axis
# contributions `axis_contrib(metric, q[a], lo[a], hi[a])` combined by an
# associative reduction `combine(metric, ·, ·)`. The descent tracks the
# combined value and, at the far-child gate, swaps one axis's contribution
# in O(1) (or O(d) for max-reductions, which are not on the safe list yet).
#
# `worst_threshold(metric, worst)` returns the cell-distance value above
# which a cell can be safely pruned. For Euclidean we square `worst` so the
# comparison is squared-vs-squared (avoids a `sqrt` in the hot path). For
# SqEuclidean / Cityblock / Minkowski(p) it equals `worst` directly (the
# rolling bound is already in those units). Chebyshev is L_inf and uses
# max-combine, requiring a different scheme — not currently supported by
# the rolling-bound path; it falls back to the legacy axis prune.

# axis_contrib: per-axis lower-bound contribution for cell with q[a] outside
# [lo, hi]. When q[a] is inside the cell, both `q - hi` and `lo - q` are
# nonpositive so `excess` is 0.
@inline function _axis_excess(q_val::S, lo::S, hi::S) where {S<:AbstractFloat}
    if q_val < lo
        return lo - q_val
    elseif q_val > hi
        return q_val - hi
    else
        return zero(S)
    end
end

# Rolling-bound contribution per axis, in the metric's prune units.
@inline _axis_contrib(::Distances.Euclidean,    e::S) where {S} = e * e
@inline _axis_contrib(::Distances.SqEuclidean,  e::S) where {S} = e * e
@inline _axis_contrib(::Distances.Cityblock,    e::S) where {S} = e
@inline function _axis_contrib(m::Distances.Minkowski, e::S) where {S}
    p = S(m.p)
    return e^p
end

# Convert the heap's `worst` (in metric-returned units) to the prune
# threshold (cell_dist >= threshold => prune).
@inline _worst_threshold(::Distances.Euclidean,   w::S) where {S} = w * w
@inline _worst_threshold(::Distances.SqEuclidean, w::S) where {S} = w
@inline _worst_threshold(::Distances.Cityblock,   w::S) where {S} = w
@inline function _worst_threshold(m::Distances.Minkowski, w::S) where {S}
    p = S(m.p)
    return w^p
end

# Whether the rolling-bound (sum-of-axis-contribs) path is the one used.
# Metrics not in this set fall back to the legacy per-axis prune in the
# query, which is correct only for `axis_distance <= worst`-compatible
# metrics (the previous safe list minus the squared / power case).
@inline _kdtree_use_rolling_bound(::Distances.Euclidean)   = true
@inline _kdtree_use_rolling_bound(::Distances.SqEuclidean) = true
@inline _kdtree_use_rolling_bound(::Distances.Cityblock)   = true
@inline _kdtree_use_rolling_bound(::Distances.Minkowski)   = true
@inline _kdtree_use_rolling_bound(_) = false

# --- Descent driver ------------------------------------------------------

@inline function _kdtree_descent!(
    index::KDTreeIndex{T,D},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    heap::BoundedMaxHeap{S},
) where {T<:LinearAlgebra.BlasFloat,D,S<:AbstractFloat}
    if _kdtree_use_rolling_bound(index.distance)
        d = index.dimension
        # Per-axis cell bounds, initialised to the unconstrained "infinite
        # cell" (q is necessarily inside, so excess is 0). Splits encountered
        # on the descent path narrow the cell; that's the only source of
        # nonzero per-axis contributions. Allocated per query — the descent
        # stack must not be shared across concurrent queries.
        cell_lo = fill(typemin(S), d)
        cell_hi = fill(typemax(S), d)
        cell_dist = zero(S)
        _search_kdtree_rolling!(index, index.root, data, q, heap,
                                cell_lo, cell_hi, cell_dist)
    else
        _search_kdtree_legacy!(index, index.root, data, q, heap)
    end
    return
end

# Rolling-bound search. `cell_dist` is the running per-axis-summed lower
# bound on `dist(q, p)` for any point `p` in the current cell, in the
# metric's prune units (see `_axis_contrib` / `_worst_threshold`). The
# routine mutates `cell_lo` / `cell_hi` along the descent and restores
# them on the way back up — the arrays are conceptually a stack but
# reused in place.
function _search_kdtree_rolling!(
    index::KDTreeIndex{T,D},
    node_id::Int,
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    heap::BoundedMaxHeap{S},
    cell_lo::Vector{S},
    cell_hi::Vector{S},
    cell_dist::S,
) where {T<:LinearAlgebra.BlasFloat,D,S<:AbstractFloat}
    node_id == 0 && return
    @inbounds node = index.nodes[node_id]

    if is_leaf(node)
        bucket_lo = node.left
        bucket_hi = node.right
        @inbounds for j in bucket_lo:bucket_hi
            point_id = index.indices[j]
            dist = S(index.distance(@view(data[:, point_id]), q))
            push!(heap, point_id, dist)
        end
        return
    end

    metric = index.distance
    axis = node.axis
    split_value = S(node.split_value)
    q_val = S(q[axis])
    go_left = q_val < split_value
    near_child = go_left ? node.left : node.right
    far_child  = go_left ? node.right : node.left

    @inbounds saved_lo = cell_lo[axis]
    @inbounds saved_hi = cell_hi[axis]
    old_axis_contrib = _axis_contrib(metric, _axis_excess(q_val, saved_lo, saved_hi))

    # --- Near child: clip cell on split axis, axis contribution does not
    # increase (q is on the near side of the split, so excess to the near
    # cell is the same as to the parent cell on that axis).
    if go_left
        @inbounds cell_hi[axis] = min(saved_hi, split_value)
    else
        @inbounds cell_lo[axis] = max(saved_lo, split_value)
    end
    new_near_contrib = _axis_contrib(metric, _axis_excess(q_val, cell_lo[axis], cell_hi[axis]))
    near_cell_dist = cell_dist - old_axis_contrib + new_near_contrib
    _search_kdtree_rolling!(index, near_child, data, q, heap,
                            cell_lo, cell_hi, near_cell_dist)
    # restore axis bounds
    @inbounds cell_lo[axis] = saved_lo
    @inbounds cell_hi[axis] = saved_hi

    # --- Far child: clip the opposite side. New axis contribution may grow
    # because q is on the wrong side of `split_value` w.r.t. this cell.
    if go_left
        @inbounds cell_lo[axis] = max(saved_lo, split_value)
    else
        @inbounds cell_hi[axis] = min(saved_hi, split_value)
    end
    @inbounds new_far_contrib = _axis_contrib(metric, _axis_excess(q_val, cell_lo[axis], cell_hi[axis]))
    far_cell_dist = cell_dist - old_axis_contrib + new_far_contrib

    not_full = length(heap) < heap.capacity
    if not_full
        _search_kdtree_rolling!(index, far_child, data, q, heap,
                                cell_lo, cell_hi, far_cell_dist)
    else
        worst = peek_max(heap).dist
        threshold = _worst_threshold(metric, worst)
        if far_cell_dist < threshold
            _search_kdtree_rolling!(index, far_child, data, q, heap,
                                    cell_lo, cell_hi, far_cell_dist)
        end
    end
    @inbounds cell_lo[axis] = saved_lo
    @inbounds cell_hi[axis] = saved_hi
    return
end

# Legacy axis-distance prune. Kept for safe metrics where the rolling
# bound has not been wired up (e.g. Chebyshev, weighted variants). The
# `axis_distance <= worst` compare is correct only for metrics where a
# per-axis projection upper-bounds the full-distance contribution from
# that axis; that's the original FBF77 safety class for non-squared
# additive metrics.
function _search_kdtree_legacy!(
    index::KDTreeIndex{T,D},
    node_id::Int,
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    heap::BoundedMaxHeap{S},
) where {T<:LinearAlgebra.BlasFloat,D,S<:AbstractFloat}
    node_id == 0 && return
    @inbounds node = index.nodes[node_id]

    if is_leaf(node)
        bucket_lo = node.left
        bucket_hi = node.right
        @inbounds for j in bucket_lo:bucket_hi
            point_id = index.indices[j]
            dist = S(index.distance(@view(data[:, point_id]), q))
            push!(heap, point_id, dist)
        end
        return
    end

    axis = node.axis
    split_value = node.split_value
    q_val = q[axis]
    go_left = q_val < split_value
    near_child = go_left ? node.left : node.right
    far_child  = go_left ? node.right : node.left

    _search_kdtree_legacy!(index, near_child, data, q, heap)

    not_full = length(heap) < heap.capacity
    worst = not_full ? typemax(S) : peek_max(heap).dist
    axis_distance = abs(q_val - split_value)
    if axis_distance <= worst || not_full
        _search_kdtree_legacy!(index, far_child, data, q, heap)
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
    _kdtree_descent!(index, data, q, heap)
    sorted = copy(heap.data)
    sort!(sorted, by = n -> (n.dist, n.id))
    return sorted
end
