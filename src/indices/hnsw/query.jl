using Base: BitSet

function query(
    index::HNSWIndex{T},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer;
    ef_search::Union{Nothing,Int} = nothing,
) where {T<:LinearAlgebra.BlasFloat}
    validate_index_dimensions(index, data, q)
    k <= 0 && return Int[]
    index.n_points == 0 && return Int[]
    base_policy = index.traversal_policy
    ef = ef_search === nothing ? default_ef(base_policy) : ef_search
    ef = max(ef, k)

    entry = NeighborCandidate(index.entry_point, index.distance(@view(data[:, index.entry_point]), q))
    visited = BitSet()
    push!(visited, entry.id)

    for layer = index.max_layer:-1:1
        entry = _greedy_descent(index, layer, entry.id, q, data)
    end
    base_results =
        _search_layer(index, 0, entry, q, data, ef; visited = visited)

    sort!(base_results, by = c -> c.dist)
    limit = min(k, length(base_results))
    return [base_results[i].id for i in 1:limit]
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
    results = Vector{Vector{Int}}(undef, n_queries)

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
    isempty(queries) && return Vector{Vector{Int}}()

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
        results = _search_layer(
            index,
            layer,
            NeighborCandidate(current, current_dist),
            point,
            data,
            index.ef_construction,
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
    visited = BitSet(),
) where {T}
    policy = with_ef(index.traversal_policy, ef)
    state = initialize_state(policy, entry)
    push!(visited, entry.id)
    adjacency = index.layers[layer + 1]

    while should_continue(policy, state)
        current = pop_pending!(policy, state)
        worst = worst_distance(policy, state)
        if current.dist > worst
            break
        end
        @inbounds for neighbor in adjacency[current.id]
            if neighbor in visited
                continue
            end
            push!(visited, neighbor)
            dist = index.distance(@view(data[:, neighbor]), q)
            if length(state.best) < policy.ef_search || dist < state.best[end].dist
                maybe_push_candidate!(policy, state, NeighborCandidate(neighbor, dist))
            end
        end
    end
    return copy(state.best)
end

function _connect_new_node!(
    index::HNSWIndex,
    layer::Int,
    node_id::Int,
    neighbors::Vector{NeighborCandidate{T}},
    data,
) where {T}
    adjacency = index.layers[layer + 1]
    adjacency[node_id] = Int[]
    for neighbor in neighbors
        _link_nodes!(index, layer, node_id, neighbor.id, data)
    end
end

function _link_nodes!(index::HNSWIndex, layer::Int, a::Int, b::Int, data)
    a == b && return
    adjacency = index.layers[layer + 1]
    list_a = adjacency[a]
    list_b = adjacency[b]

    # Add neighbors without checking for duplicates (O(1) instead of O(M))
    # Pruning will naturally handle any duplicates that arise
    push!(list_a, b)
    _prune_list!(index, list_a, a, data, max_degree(index.neighbor_policy))

    push!(list_b, a)
    _prune_list!(index, list_b, b, data, max_degree(index.neighbor_policy))

    # No need to reassign: list_a and list_b are references to adjacency lists,
    # already modified in-place by push! and _prune_list!
end

function _prune_list!(index::HNSWIndex, list::Vector{Int}, center::Int, data, limit::Int)
    length(list) <= limit && return
    center_vec = @view(data[:, center])

    # Remove duplicates that may have been added by _link_nodes!
    unique!(list)
    length(list) <= limit && return

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
    pruned = select_neighbors(
        index.neighbor_policy,
        candidates,
        data,
        index.distance;
        limit = limit,
    )
    resize!(list, length(pruned))
    @inbounds for (i, cand) in enumerate(pruned)
        list[i] = cand.id
    end
end

function _ensure_layers!(index::HNSWIndex, level::Int)
    required = level + 1
    while length(index.layers) < required
        push!(index.layers, [Int[] for _ in 1:index.n_points])
    end
end

function _ensure_node_slots!(index::HNSWIndex, node_id::Int)
    for layer in index.layers
        while length(layer) < node_id
            push!(layer, Int[])
        end
    end
end
