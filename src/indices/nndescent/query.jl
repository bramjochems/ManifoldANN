using LinearAlgebra
using Random

function query(
    index::NNDescentIndex{T},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer;
    ef_search::Union{Nothing,Integer} = nothing,
    bounded_candidates::Union{Nothing,Integer} = nothing,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:LinearAlgebra.BlasFloat}
    validate_index_dimensions(index, data, q)
    S = float(T)
    k <= 0 && return Neighbor{S}[]
    actual_k = min(Int(k), index.n_points)
    actual_k == 0 && return Neighbor{S}[]

    beam = ef_search === nothing ? max(actual_k, index.k) : max(actual_k, Int(ef_search))
    beam > 0 || (beam = actual_k)
    cand_cap = bounded_candidates === nothing ? typemax(Int) : max(Int(bounded_candidates), actual_k)

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
            # bounded_candidates: NND.jl-style frontier prune. When the result
            # buffer is already at the candidate cap and this neighbor cannot
            # improve it, skip the push entirely. Default (cand_cap=typemax)
            # is bit-equal to the unbounded path.
            if length(best) >= cand_cap && neighbor_dist >= best[end].dist
                continue
            end
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

"""
    NNDescentBatchScratch{S}

Per-worker scratch state for the NN-Descent BATCH query path. Reused across
all queries one worker processes, so its allocation cost amortises over many
calls.

Holds:
- a `BitVector` visited mask (sized to `n_points`)
- backing `Vector{NeighborCandidate{S}}` for the pending min-heap
- `best` result buffer (`Vector{NeighborCandidate{S}}`)
- entry-point selection buffers (`Set{Int}` + `Vector{Int}`)

Lifetime is owned by one task. Concurrent use of the same `NNDescentBatchScratch`
from multiple threads is undefined behaviour.

NOT used by the single-query API — that path allocates fresh buffers per call
to preserve the existing thread-safety contract (single-query callers may share
an index across threads).
"""
struct NNDescentBatchScratch{S<:AbstractFloat}
    visited::BitVector
    pending_data::Vector{NeighborCandidate{S}}
    best::Vector{NeighborCandidate{S}}
    entry_seen::Set{Int}
    entry_selected::Vector{Int}
end

function NNDescentBatchScratch{S}(n_points::Int, beam_hint::Int) where {S<:AbstractFloat}
    pending = Vector{NeighborCandidate{S}}()
    best    = Vector{NeighborCandidate{S}}()
    sizehint!(pending, beam_hint)
    sizehint!(best, beam_hint)
    return NNDescentBatchScratch{S}(
        falses(n_points),
        pending,
        best,
        Set{Int}(),
        Int[],
    )
end

# Run one query against `index` using caller-owned scratch buffers. NOT
# re-entrant — caller must guarantee exclusive ownership of `scratch`.
function _query_pooled!(
    scratch::NNDescentBatchScratch{S},
    index::NNDescentIndex{T},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer,
    ef_search::Union{Nothing,Integer},
    rng::AbstractRNG,
    bounded_candidates::Union{Nothing,Integer} = nothing,
) where {T<:LinearAlgebra.BlasFloat,S<:AbstractFloat}
    validate_index_dimensions(index, data, q)
    k <= 0 && return Neighbor{S}[]
    actual_k = min(Int(k), index.n_points)
    actual_k == 0 && return Neighbor{S}[]

    beam = ef_search === nothing ? max(actual_k, index.k) : max(actual_k, Int(ef_search))
    beam > 0 || (beam = actual_k)
    cand_cap = bounded_candidates === nothing ? typemax(Int) : max(Int(bounded_candidates), actual_k)

    fill!(scratch.visited, false)
    empty!(scratch.pending_data)
    empty!(scratch.best)

    start_ids = _pick_entry_points_pooled!(
        scratch.entry_selected, scratch.entry_seen,
        index.n_points, min(index.k, beam), rng,
    )

    first_id = start_ids[1]
    first_dist = S(index.distance(view(data, :, first_id), q))

    candidates = NeighborMinHeap{S}(scratch.pending_data)
    push!(candidates, NeighborCandidate{S}(first_id, first_dist))
    @inbounds scratch.visited[first_id] = true

    @inbounds for idx in 2:length(start_ids)
        id = start_ids[idx]
        scratch.visited[id] = true
        dist = S(index.distance(view(data, :, id), q))
        push!(candidates, NeighborCandidate{S}(id, dist))
    end

    best = scratch.best
    while !isempty(candidates)
        current = popfirst!(candidates)
        if length(best) >= beam && current.dist > best[end].dist
            break
        end
        _insert_best_neighbor!(best, current, beam)
        @inbounds for neighbor_id in index.neighbors[current.id]
            scratch.visited[neighbor_id] && continue
            scratch.visited[neighbor_id] = true
            neighbor_dist = S(index.distance(view(data, :, neighbor_id), q))
            if length(best) >= cand_cap && neighbor_dist >= best[end].dist
                continue
            end
            push!(candidates, NeighborCandidate{S}(neighbor_id, neighbor_dist))
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

# Worker body is a top-level function so each `Threads.@spawn` call site
# allocates its own `scratch` from a clean stack frame, with no closure
# capture across the worker-spawn loop. Keeping the structure explicit
# avoids subtle aliasing issues that can arise when a `let scratch = ...`
# block is used inside the spawn loop.
function _nndescent_batch_worker!(
    results::Vector{Vector{Neighbor{S}}},
    work::Channel{Int},
    index::NNDescentIndex{T},
    data::AbstractMatrix{T},
    queries::AbstractMatrix{T},
    k::Integer,
    ef_search::Union{Nothing,Integer},
    parent_seed::UInt64,
    ::Type{S},
    beam_hint::Int,
    bounded_candidates::Union{Nothing,Integer},
) where {T<:LinearAlgebra.BlasFloat,S<:AbstractFloat}
    scratch = NNDescentBatchScratch{S}(index.n_points, beam_hint)
    for i in work
        results[i] = _query_pooled!(
            scratch, index, data, view(queries, :, i),
            k, ef_search, query_child_rng(parent_seed, i),
            bounded_candidates,
        )
    end
    return nothing
end

function query(
    index::NNDescentIndex{T},
    data::AbstractMatrix{T},
    queries::AbstractMatrix{T},
    k::Integer;
    ef_search::Union{Nothing,Integer} = nothing,
    bounded_candidates::Union{Nothing,Integer} = nothing,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:LinearAlgebra.BlasFloat}
    validate_index_query_matrix(index, queries)
    S = float(T)
    n_queries = size(queries, 2)
    n_queries == 0 && return Vector{Vector{Neighbor{S}}}()
    parent_seed = derive_child_seed(rng)
    results = Vector{Vector{Neighbor{S}}}(undef, n_queries)

    actual_k = min(Int(k), index.n_points)
    if k <= 0 || actual_k == 0
        @inbounds for i in 1:n_queries
            results[i] = Neighbor{S}[]
        end
        return results
    end
    beam_hint = ef_search === nothing ? max(actual_k, index.k) : max(actual_k, Int(ef_search))

    if Threads.nthreads() == 1 || n_queries < BATCH_THREAD_THRESHOLD
        scratch = NNDescentBatchScratch{S}(index.n_points, beam_hint)
        @inbounds for i in 1:n_queries
            results[i] = _query_pooled!(
                scratch, index, data, view(queries, :, i), k, ef_search,
                query_child_rng(parent_seed, i),
                bounded_candidates,
            )
        end
        return results
    end

    nworkers = Threads.nthreads()
    work = Channel{Int}(min(n_queries, max(64, 4 * nworkers)); spawn=true) do ch
        @inbounds for i in 1:n_queries
            put!(ch, i)
        end
    end

    workers = Vector{Task}(undef, nworkers)
    for w in 1:nworkers
        workers[w] = Threads.@spawn _nndescent_batch_worker!(
            results, work, index, data, queries, k, ef_search, parent_seed,
            S, beam_hint, bounded_candidates,
        )
    end
    foreach(wait, workers)
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

# Pooled variant: writes into caller-owned `selected` Vector and `seen` Set,
# both of which are emptied first. Returns `selected` for convenience.
function _pick_entry_points_pooled!(
    selected::Vector{Int},
    seen::Set{Int},
    n_points::Int,
    count::Int,
    rng::AbstractRNG,
)
    count = max(count, 1)
    count = min(count, n_points)
    empty!(selected)
    empty!(seen)
    while length(selected) < count
        candidate = rand(rng, 1:n_points)
        candidate in seen && continue
        push!(selected, candidate)
        push!(seen, candidate)
    end
    return selected
end

_pick_entry_points(n_points::Int, count::Int, rng::AbstractRNG) =
    _pick_entry_points_pooled!(Int[], Set{Int}(), n_points, count, rng)

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
