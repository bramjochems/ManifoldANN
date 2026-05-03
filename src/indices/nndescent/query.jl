using LinearAlgebra
using Random

function query(
    index::NNDescentIndex{T},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer;
    ef_search::Union{Nothing,Integer} = nothing,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:LinearAlgebra.BlasFloat}
    validate_index_dimensions(index, data, q)
    S = float(T)
    k <= 0 && return Neighbor{S}[]
    actual_k = min(Int(k), index.n_points)
    actual_k == 0 && return Neighbor{S}[]

    beam = ef_search === nothing ? max(actual_k, index.k) : max(actual_k, Int(ef_search))
    beam > 0 || (beam = actual_k)

    start_ids = _pick_entry_points(index.n_points, min(index.k, beam), rng)
    first_id = start_ids[1]
    first_dist = index.distance(view(data, :, first_id), q)
    dist_type = typeof(first_dist)
    dist_type <: AbstractFloat ||
        throw(
            ArgumentError(
                "distance function must return an AbstractFloat, got $(dist_type)",
            ),
        )

    candidates = NeighborMinHeap(NeighborCandidate{dist_type}(first_id, first_dist))
    visited = falses(index.n_points)
    best = Vector{NeighborCandidate{dist_type}}()

    # Seed heap with remaining entry points
    visited[first_id] = true
    @inbounds for idx in 2:length(start_ids)
        id = start_ids[idx]
        visited[id] = true
        dist = index.distance(view(data, :, id), q)
        push!(candidates, NeighborCandidate{dist_type}(id, dist))
    end

    while !isempty(candidates)
        current = popfirst!(candidates)
        if length(best) >= beam && current.dist > best[end].dist
            break
        end
        _insert_best_neighbor!(best, current, beam)
        for neighbor_id in index.neighbors[current.id]
            visited[neighbor_id] && continue
            visited[neighbor_id] = true
            neighbor_dist = index.distance(view(data, :, neighbor_id), q)
            push!(
                candidates,
                NeighborCandidate{dist_type}(neighbor_id, neighbor_dist),
            )
        end
    end

    result_count = min(actual_k, length(best))
    neighbors = Vector{Neighbor{S}}(undef, result_count)
    @inbounds for i in 1:result_count
        candidate = best[i]
        neighbors[i] = Neighbor{S}(candidate.id, S(candidate.dist))
    end
    return neighbors
end

function query(
    index::NNDescentIndex{T},
    data::AbstractMatrix{T},
    queries::AbstractMatrix{T},
    k::Integer;
    ef_search::Union{Nothing,Integer} = nothing,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:LinearAlgebra.BlasFloat}
    size(queries, 1) == index.dimension ||
        throw(DimensionMismatch("Expected queries with $(index.dimension) rows"))
    n_queries = size(queries, 2)
    n_queries == 0 && return Vector{Vector{Neighbor{float(T)}}}()
    # Spawn deterministic per-query child RNGs up front so threaded execution
    # stays reproducible (each task gets its own RNG; no shared mutable state).
    child_rngs = spawn_child_rngs(rng, n_queries)
    results = Vector{Vector{Neighbor{float(T)}}}(undef, n_queries)
    if Threads.nthreads() == 1 || n_queries < BATCH_THREAD_THRESHOLD
        @inbounds for i in 1:n_queries
            results[i] = query(
                index,
                data,
                view(queries, :, i),
                k;
                ef_search = ef_search,
                rng = child_rngs[i],
            )
        end
    else
        Threads.@threads for i in 1:n_queries
            results[i] = query(
                index,
                data,
                view(queries, :, i),
                k;
                ef_search = ef_search,
                rng = child_rngs[i],
            )
        end
    end
    return results
end

function query(
    index::NNDescentIndex{T},
    data::AbstractMatrix{T},
    queries::Vector{<:AbstractVector{T}},
    k::Integer;
    kwargs...,
) where {T<:LinearAlgebra.BlasFloat}
    isempty(queries) && return Vector{Vector{Neighbor{float(T)}}}()
    matrix = reduce(hcat, queries)
    return query(index, data, matrix, k; kwargs...)
end

function _pick_entry_points(n_points::Int, count::Int, rng::AbstractRNG)
    count = max(count, 1)
    count = min(count, n_points)
    selected = Int[]
    seen = Set{Int}()
    while length(selected) < count
        candidate = rand(rng, 1:n_points)
        candidate in seen && continue
        push!(selected, candidate)
        push!(seen, candidate)
    end
    return selected
end

function _insert_best_neighbor!(
    list::Vector{NeighborCandidate{T}},
    candidate::NeighborCandidate{T},
    limit::Int,
) where {T}
    pos = 1
    len = length(list)
    @inbounds while pos <= len && list[pos].dist <= candidate.dist
        pos += 1
    end
    insert!(list, pos, candidate)
    if length(list) > limit
        pop!(list)
    end
    return nothing
end

function materialize_graph(index::NNDescentIndex)
    adjacency = [copy(neigh) for neigh in index.neighbors]
    realized_k = isempty(adjacency) ? 0 : maximum(length.(adjacency))
    return KNNGraph(adjacency, realized_k, false, nothing)
end
