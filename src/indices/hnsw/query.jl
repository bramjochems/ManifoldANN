using Base: BitSet

function query(
    index::HNSWIndex{T},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer;
    ef_search::Union{Nothing,Int} = nothing,
    distance::Function = default_distance,
) where {T<:LinearAlgebra.BlasFloat}
    validate_index_dimensions(index, data, q)
    k <= 0 && return Int[]
    index.n_points == 0 && return Int[]
    base_policy = index.traversal_policy
    ef = ef_search === nothing ? default_ef(base_policy) : ef_search
    ef = max(ef, k)

    entry = NeighborCandidate(index.entry_point, distance(@view(data[:, index.entry_point]), q))
    visited = BitSet()
    push!(visited, entry.id)

    for layer = index.max_layer:-1:1
        entry = _greedy_descent(index, layer, entry.id, q, data, distance)
    end
    base_results =
        _search_layer(index, 0, entry, q, data, ef, distance; visited = visited)

    sort!(base_results, by = c -> c.dist)
    limit = min(k, length(base_results))
    return [base_results[i].id for i in 1:limit]
end

function insert!(
    index::HNSWIndex{T},
    data::AbstractMatrix{T},
    point::AbstractVector{T};
    point_id::Union{Nothing,Int} = nothing,
    rng::AbstractRNG = Random.default_rng(),
    distance::Function = default_distance,
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
    current_dist = distance(@view(data[:, current]), point)
    for layer = index.max_layer:-1:max(level + 1, 1)
        candidate = _greedy_descent(index, layer, current, point, data, distance)
        current = candidate.id
        current_dist = candidate.dist
    end

    for layer = min(level, index.max_layer):-1:0
        results = _search_layer(index, layer, NeighborCandidate(current, current_dist), point, data, index.ef_construction, distance)
        neighbors = select_neighbors(index.neighbor_policy, results)
        _connect_new_node!(index, layer, node_id, neighbors, data, distance)
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

function _greedy_descent(index::HNSWIndex, layer::Int, entry_id::Int, q, data, distance)
    current = entry_id
    current_dist = distance(@view(data[:, current]), q)
    improved = true
    while improved
        improved = false
        @inbounds for neighbor in index.layers[layer + 1][current]
            dist = distance(@view(data[:, neighbor]), q)
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
    entry::NeighborCandidate,
    q,
    data,
    ef::Int,
    distance;
    visited = BitSet(),
)
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
            dist = distance(@view(data[:, neighbor]), q)
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
    neighbors::Vector{NeighborCandidate},
    data,
    distance,
)
    adjacency = index.layers[layer + 1]
    adjacency[node_id] = Int[]
    for neighbor in neighbors
        _link_nodes!(index, layer, node_id, neighbor.id, data, distance)
    end
end

function _link_nodes!(index::HNSWIndex, layer::Int, a::Int, b::Int, data, distance)
    a == b && return
    adjacency = index.layers[layer + 1]
    list_a = adjacency[a]
    list_b = adjacency[b]
    if !(b in list_a)
        push!(list_a, b)
        _prune_list!(list_a, a, data, distance, max_degree(index.neighbor_policy))
    end
    if !(a in list_b)
        push!(list_b, a)
        _prune_list!(list_b, b, data, distance, max_degree(index.neighbor_policy))
    end
    adjacency[a] = list_a
    adjacency[b] = list_b
end

function _prune_list!(list::Vector{Int}, center::Int, data, distance, limit::Int)
    length(list) <= limit && return
    center_vec = @view(data[:, center])
    sort!(list, by = id -> distance(center_vec, @view(data[:, id])))
    resize!(list, limit)
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
