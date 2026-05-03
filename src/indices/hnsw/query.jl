using Base: BitSet

# Visited-set abstraction. Two implementations:
#   * StampVisited: O(1) reset via a generation counter, used during build
#     where the buffer can be reused across all insert!/_search_layer calls.
#     Not thread-safe — single-build only.
#   * BitSetVisited: per-call BitSet, used by `query` (potentially concurrent).
struct StampVisited
    stamps::Vector{UInt32}
    generation::UInt32
end

@inline _was_visited(v::StampVisited, id::Int) = (@inbounds v.stamps[id] == v.generation)
@inline function _mark_visited!(v::StampVisited, id::Int)
    @inbounds v.stamps[id] = v.generation
    return
end

struct BitSetVisited
    set::BitSet
end
BitSetVisited() = BitSetVisited(BitSet())

@inline _was_visited(v::BitSetVisited, id::Int) = id in v.set
@inline _mark_visited!(v::BitSetVisited, id::Int) = (push!(v.set, id); nothing)

# Acquire a build-time visited buffer. Bumps the generation; on wrap,
# zeroes the buffer. Caller must ensure stamps is sized to n_points.
function _acquire_build_visited!(index::HNSWIndex)
    # Size to n_points + 1: defensive against any future call path that ever
    # marks the in-flight new node (currently it doesn't, because the new node
    # has no incoming edges yet at search time, but losing that invariant
    # silently corrupts memory via @inbounds writes).
    n = index.n_points + 1
    if length(index.visit_stamps) < n
        old = length(index.visit_stamps)
        resize!(index.visit_stamps, max(n, 2 * old, 16))
        @inbounds for i in (old+1):length(index.visit_stamps)
            index.visit_stamps[i] = UInt32(0)
        end
    end
    gen = index.visit_generation + UInt32(1)
    if gen == UInt32(0)
        # Wrapped — zero buffer and start at 1.
        fill!(index.visit_stamps, UInt32(0))
        gen = UInt32(1)
    end
    index.visit_generation = gen
    return StampVisited(index.visit_stamps, gen)
end

function query(
    index::HNSWIndex{T},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer;
    ef_search::Union{Nothing,Int} = nothing,
) where {T<:LinearAlgebra.BlasFloat}
    validate_index_dimensions(index, data, q)
    S = float(T)
    k <= 0 && return Neighbor{S}[]
    index.n_points == 0 && return Neighbor{S}[]
    base_policy = index.traversal_policy
    ef = ef_search === nothing ? default_ef(base_policy) : ef_search
    ef = max(ef, k)

    entry = NeighborCandidate(index.entry_point, index.distance(@view(data[:, index.entry_point]), q))
    visited = BitSetVisited()
    _mark_visited!(visited, entry.id)

    for layer = index.max_layer:-1:1
        entry = _greedy_descent(index, layer, entry.id, q, data)
    end
    base_results =
        _search_layer(index, 0, entry, q, data, ef; visited = visited)

    sort!(base_results, by = c -> c.dist)
    limit = min(k, length(base_results))
    neighbors = Vector{Neighbor{S}}(undef, limit)
    @inbounds for i in 1:limit
        candidate = base_results[i]
        neighbors[i] = Neighbor{S}(candidate.id, S(candidate.dist))
    end
    return neighbors
end

"""
    query(index::HNSWIndex, data::Matrix, queries::Matrix, k::Integer; kwargs...)

Batch query interface: process multiple queries efficiently.

# Arguments
- `index`: The HNSW index
- `data`: Data matrix (dimension × n_points)
- `queries`: Query matrix (dimension × n_queries)
- `k`: Number of neighbors to return per query
- `kwargs...`: Additional arguments passed to single query (e.g., ef_search, distance)

# Returns
Vector of result vectors, one per query. Each result is a vector of neighbor indices.

# Performance
This method processes queries in parallel using all available threads (`Threads.@threads`).
Each query is independent and reads only from the index, making parallelization safe.
When called from Python via juliacall, this minimizes the Python↔Julia bridge overhead
by crossing the language boundary only once while distributing work across cores.
"""
function query(
    index::HNSWIndex{T},
    data::AbstractMatrix{T},
    queries::AbstractMatrix{T},
    k::Integer;
    kwargs...
) where {T<:LinearAlgebra.BlasFloat}
    size(queries, 1) == index.dimension ||
        throw(DimensionMismatch("Expected queries with $(index.dimension) rows"))

    n_queries = size(queries, 2)
    results = Vector{Vector{Neighbor{float(T)}}}(undef, n_queries)

    Threads.@threads for i in 1:n_queries
        q = @view queries[:, i]
        results[i] = query(index, data, q, k; kwargs...)
    end

    return results
end

"""
    query(index::HNSWIndex, data::Matrix, queries::Vector{<:Vector}, k::Integer; kwargs...)

Convenience batch query interface using a vector of query vectors.

# Note
This method converts the Vector{Vector} to a Matrix internally for performance.
For large batches, prefer passing queries as a Matrix directly.
"""
function query(
    index::HNSWIndex{T},
    data::AbstractMatrix{T},
    queries::Vector{<:AbstractVector{T}},
    k::Integer;
    kwargs...
) where {T<:LinearAlgebra.BlasFloat}
    isempty(queries) && return Vector{Vector{Neighbor{float(T)}}}()

    # Stack queries into a matrix (dimension × n_queries)
    queries_mat = reduce(hcat, queries)

    return query(index, data, queries_mat, k; kwargs...)
end

function insert!(
    index::HNSWIndex{T},
    data::AbstractMatrix{T},
    point::AbstractVector{T};
    point_id::Union{Nothing,Int} = nothing,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:LinearAlgebra.BlasFloat}
    size(data, 1) == index.dimension ||
        throw(DimensionMismatch("Expected data with $(index.dimension) rows"))
    length(point) == index.dimension ||
        throw(DimensionMismatch("Expected point dimension $(index.dimension)"))

    node_id = point_id === nothing ? index.n_points + 1 : point_id
    node_id == index.n_points + 1 ||
        throw(ArgumentError("Points must be inserted sequentially (next id = $(index.n_points + 1))"))
    size(data, 2) >= node_id ||
        throw(ArgumentError("Data matrix must contain at least $node_id columns"))

    level = sample_layer(index.planner, rng)
    _ensure_layers!(index, level)
    _ensure_node_slots!(index, node_id)

    if index.n_points == 0
        index.entry_point = node_id
        index.max_layer = level
        index.n_points = 1
        return index
    end

    current = index.entry_point
    current_dist = index.distance(@view(data[:, current]), point)
    for layer = index.max_layer:-1:max(level + 1, 1)
        candidate = _greedy_descent(index, layer, current, point, data)
        current = candidate.id
        current_dist = candidate.dist
    end

    for layer = min(level, index.max_layer):-1:0
        visited = _acquire_build_visited!(index)
        results = _search_layer(
            index,
            layer,
            NeighborCandidate(current, current_dist),
            point,
            data,
            index.ef_construction;
            visited = visited,
        )
        neighbors = select_neighbors(index.neighbor_policy, results, data, index.distance)
        _connect_new_node!(index, layer, node_id, neighbors, data)
        current = isempty(neighbors) ? current : neighbors[1].id
        current_dist = isempty(neighbors) ? current_dist : neighbors[1].dist
    end

    if level > index.max_layer
        index.max_layer = level
        index.entry_point = node_id
    end

    index.n_points += 1
    return index
end

function _greedy_descent(index::HNSWIndex, layer::Int, entry_id::Int, q, data)
    current = entry_id
    current_dist = index.distance(@view(data[:, current]), q)
    improved = true
    while improved
        improved = false
        @inbounds for neighbor in index.layers[layer + 1][current]
            dist = index.distance(@view(data[:, neighbor]), q)
            if dist < current_dist
                current = neighbor
                current_dist = dist
                improved = true
            end
        end
    end
    return NeighborCandidate(current, current_dist)
end

function _search_layer(
    index::HNSWIndex,
    layer::Int,
    entry::NeighborCandidate{T},
    q,
    data,
    ef::Int;
    visited = BitSetVisited(),
) where {T}
    policy = with_ef(index.traversal_policy, ef)
    state = initialize_state(policy, entry)
    _mark_visited!(visited, entry.id)
    adjacency = index.layers[layer + 1]

    while should_continue(policy, state)
        current = pop_pending!(policy, state)
        worst = worst_distance(policy, state)
        if current.dist > worst
            break
        end
        @inbounds for neighbor in adjacency[current.id]
            if _was_visited(visited, neighbor)
                continue
            end
            _mark_visited!(visited, neighbor)
            dist = index.distance(@view(data[:, neighbor]), q)
            if length(state.best) < policy.ef_search || dist < worst_distance(state.best)
                maybe_push_candidate!(policy, state, NeighborCandidate(neighbor, dist))
            end
        end
    end
    # CONTRACT: caller takes ownership of the returned vector (= heap's backing
    # buffer). Heap is dead after this point. Buffer is sorted ascending by
    # distance — the build path's `select_neighbors` short-circuit at
    # neighbor_policy.jl line ~71 relies on `neighbors[1]` being the nearest.
    # MergeSort pinned for graph-signature stability across Julia versions.
    sort!(state.best.data, alg = Base.Sort.MergeSort, by = c -> c.dist)
    return state.best.data
end

function _connect_new_node!(
    index::HNSWIndex,
    layer::Int,
    node_id::Int,
    neighbors::Vector{NeighborCandidate{T}},
    data,
) where {T}
    adjacency = index.layers[layer + 1]

    # Write neighbor ids into the pre-sized adjacency slot in place. It already
    # has capacity max_degree+1 from _ensure_node_slots! / _ensure_layers!.
    own_list = adjacency[node_id]
    resize!(own_list, length(neighbors))
    @inbounds for (i, n) in enumerate(neighbors)
        own_list[i] = n.id
    end

    # Update reverse edges.
    for neighbor in neighbors
        list_b = adjacency[neighbor.id]
        push!(list_b, node_id)
        _prune_list!(index, list_b, neighbor.id, data, max_degree(index.neighbor_policy))
    end
end

function _prune_list!(index::HNSWIndex, list::Vector{Int}, center::Int, data, limit::Int)
    length(list) <= limit && return

    center_vec = @view(data[:, center])

    first_id = list[1]
    first_dist = index.distance(center_vec, @view(data[:, first_id]))
    TDist = typeof(first_dist)
    candidates = Vector{NeighborCandidate{TDist}}(undef, length(list))
    candidates[1] = NeighborCandidate(first_id, first_dist)
    @inbounds for i in 2:length(list)
        id = list[i]
        dist = index.distance(center_vec, @view(data[:, id]))
        candidates[i] = NeighborCandidate(id, dist)
    end
    n_kept = _select_into!(
        index.neighbor_policy,
        list,
        candidates,
        data,
        index.distance,
        limit,
    )
    resize!(list, n_kept)
    return
end

# In-place select: writes kept neighbor ids straight into `out_ids[1:n_kept]`,
# returns n_kept. Avoids the per-call `selected` Vector and `selected_ids`
# BitSet that the generic select_neighbors allocates.

# Note: _select_into! does NOT short-circuit on `length(candidates) <= cap`
# the way `select_neighbors(::DiversifiedNeighborPolicy)` does — at the prune
# call site the caller already guards on `length(list) > limit`, so the
# short-circuit branch is unreachable. The two implementations are therefore
# NOT general-purpose interchangeable.
function _select_into!(
    policy::HeuristicNeighborPolicy,
    out_ids::Vector{Int},
    candidates::Vector{NeighborCandidate{T}},
    ::AbstractMatrix,
    ::Function,
    limit::Int,
) where {T}
    cap = min(limit, length(candidates))
    cap == 0 && return 0
    # partialsort! is not stable; tie-breaking is input-order-dependent. This
    # mirrors the legacy `select_neighbors(::HeuristicNeighborPolicy)` path.
    partialsort!(candidates, 1:cap, by = c -> c.dist)
    @inbounds for i in 1:cap
        out_ids[i] = candidates[i].id
    end
    return cap
end

function _select_into!(
    policy::DiversifiedNeighborPolicy,
    out_ids::Vector{Int},
    candidates::Vector{NeighborCandidate{T}},
    data::AbstractMatrix,
    distance_fn,
    limit::Int,
) where {T}
    n = length(candidates)
    n == 0 && return 0
    cap = min(limit, n)
    # MergeSort pinned for graph-signature stability across Julia versions.
    sort!(candidates, alg = Base.Sort.MergeSort, by = c -> c.dist)

    n_kept = 0
    @inbounds for ci in 1:n
        cand = candidates[ci]
        dominated = false
        # Linear scan over kept prefix (n_kept ≤ cap = M, typically ≤ 16).
        for j in 1:n_kept
            chosen_id = out_ids[j]
            d_bc = distance_fn(@view(data[:, cand.id]), @view(data[:, chosen_id]))
            if d_bc < cand.dist
                dominated = true
                break
            end
        end
        if !dominated
            n_kept += 1
            out_ids[n_kept] = cand.id
            n_kept == cap && break
        end
    end

    # keepPrunedConnections fallback: fill remaining slots from rejected
    # candidates in distance order, skipping already-selected ids.
    if n_kept < cap
        @inbounds for ci in 1:n
            id = candidates[ci].id
            already = false
            for j in 1:n_kept
                if out_ids[j] == id
                    already = true
                    break
                end
            end
            already && continue
            n_kept += 1
            out_ids[n_kept] = id
            n_kept == cap && break
        end
    end
    return n_kept
end

function _ensure_layers!(index::HNSWIndex, level::Int)
    required = level + 1
    cap = max_degree(index.neighbor_policy) + 1
    while length(index.layers) < required
        new_layer = Vector{NeighborList}(undef, index.n_points)
        @inbounds for i in 1:index.n_points
            v = Int[]
            sizehint!(v, cap)
            new_layer[i] = v
        end
        push!(index.layers, new_layer)
    end
end

function _ensure_node_slots!(index::HNSWIndex, node_id::Int)
    cap = max_degree(index.neighbor_policy) + 1
    for layer in index.layers
        while length(layer) < node_id
            v = Int[]
            sizehint!(v, cap)
            push!(layer, v)
        end
    end
end
