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

"""
    BatchQueryScratch{T}

Per-worker scratch state for the BATCH query path. Reused across all queries
that one worker processes, so its allocation cost amortises over many calls.

Holds:
- a generation-stamped visited buffer (sized to `n_points`)
- backing `Vector{NeighborCandidate{T}}` for the `best` (max-)heap
- backing `Vector{NeighborCandidate{T}}` for the `pending` (min-)heap

Lifetime is owned by one task. Concurrent use from multiple threads on the
same `BatchQueryScratch` is undefined behaviour.

NOT used by the single-query API — that path allocates a `BitSetVisited`
(small) plus fresh empty heap buffers, matching the pre-pooling allocation
profile. The stamp buffer here is `n_points × 4 bytes`, which is wasteful
for a one-shot query but cheap when amortised across hundreds of queries.
"""
mutable struct BatchQueryScratch{T<:AbstractFloat}
    visit_stamps::Vector{UInt32}
    visit_generation::UInt32
    best_data::Vector{NeighborCandidate{T}}
    pending_data::Vector{NeighborCandidate{T}}
end

function BatchQueryScratch{T}(n_points::Int, ef_capacity::Int) where {T<:AbstractFloat}
    best  = Vector{NeighborCandidate{T}}()
    pend  = Vector{NeighborCandidate{T}}()
    sizehint!(best, ef_capacity)
    sizehint!(pend, ef_capacity)
    return BatchQueryScratch{T}(zeros(UInt32, n_points), UInt32(0), best, pend)
end

# Bump the visit generation; on UInt32 wrap, zero the stamp buffer.
@inline function _bump_visit_generation!(scratch::BatchQueryScratch)
    g = scratch.visit_generation + UInt32(1)
    if g == UInt32(0)
        fill!(scratch.visit_stamps, UInt32(0))
        g = UInt32(1)
    end
    scratch.visit_generation = g
    return g
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

    # Single-query path: small BitSet for visited (only marks ~ef·M·log(n)
    # ids per query, so it stays tiny) + fresh empty heap buffers. This
    # matches the allocation profile from before the worker-pool refactor;
    # the heavier StampVisited buffer is only used in the batch path where
    # it amortises across many queries per worker.
    visited = BitSetVisited()
    return _query_search(index, data, q, k, ef, visited,
                         NeighborCandidate{S}[], NeighborCandidate{S}[])
end

# Core search procedure. Takes a visited set + caller-owned heap data
# buffers. The buffers may be empty (single-query path: fresh allocation,
# zero-cost `empty!`) or pre-sized and reused (batch path: ef-sized,
# `empty!` preserves capacity).
#
# NOT re-entrant — caller must guarantee exclusive ownership of all four
# scratch arguments for the duration of the call.
function _query_search(
    index::HNSWIndex{T},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer,
    ef::Int,
    visited,
    best_buf::Vector{NeighborCandidate{S}},
    pending_buf::Vector{NeighborCandidate{S}},
) where {T<:LinearAlgebra.BlasFloat, S}
    entry = NeighborCandidate(index.entry_point, index.distance(@view(data[:, index.entry_point]), q))
    _mark_visited!(visited, entry.id)

    for layer = index.max_layer:-1:1
        entry = _greedy_descent(index, layer, entry.id, q, data)
    end
    base_results = _search_layer_pooled(
        index, 0, entry, q, data, ef, visited, best_buf, pending_buf,
    )

    sort!(base_results, by = c -> c.dist)
    limit = min(k, length(base_results))
    neighbors = Vector{Neighbor{S}}(undef, limit)
    @inbounds for i in 1:limit
        candidate = base_results[i]
        neighbors[i] = Neighbor{S}(candidate.id, S(candidate.dist))
    end
    return neighbors
end

# Acquire a fresh StampVisited from a BatchQueryScratch by bumping the gen.
@inline function _acquire_batch_visited!(scratch::BatchQueryScratch, n_points::Int)
    if length(scratch.visit_stamps) < n_points
        old = length(scratch.visit_stamps)
        resize!(scratch.visit_stamps, n_points)
        @inbounds for i in (old+1):length(scratch.visit_stamps)
            scratch.visit_stamps[i] = UInt32(0)
        end
    end
    gen = _bump_visit_generation!(scratch)
    return StampVisited(scratch.visit_stamps, gen)
end

"""
    query(index::HNSWIndex, data::Matrix, queries::Matrix, k::Integer; kwargs...)

Batch query interface: process multiple queries efficiently.

# Arguments
- `index`: The HNSW index
- `data`: Data matrix (dimension × n_points)
- `queries`: Query matrix (dimension × n_queries)
- `k`: Number of neighbors to return per query
- `kwargs...`: Additional arguments passed to single query (e.g., `ef_search`)

# Returns
Vector of result vectors, one per query.

# Threading
Uses a worker pool: spawns `Threads.nthreads()` long-lived tasks; each owns a
`BatchQueryScratch` (stamp-visited buffer + heap data) reused across all
queries that worker processes. Work is distributed via a `Channel{Int}`,
giving load-balanced scheduling rather than the static partition
`Threads.@threads` gives.

The single-query API allocates fresh `BitSetVisited` + heap buffers per call
(the buffers are small for one query). The batch path's `BatchQueryScratch`
trades one larger up-front allocation per worker for amortised reuse across
many queries — net allocation per query drops by 25-30× at typical n.

# Thread-safety contract
The user-supplied `index.distance` callable MUST be safe to call concurrently
from multiple threads. The default `default_distance` and Distances.jl
metrics satisfy this. A stateful distance functor (e.g. one with internal
caching that mutates on call) does NOT — this was already true for the old
`Threads.@threads` batch path; the contract is unchanged.

The index itself MUST NOT be mutated (via `insert!`) concurrently with this
call. Concurrent reads from multiple workers are safe; concurrent
read+write is not.
"""
function query(
    index::HNSWIndex{T},
    data::AbstractMatrix{T},
    queries::AbstractMatrix{T},
    k::Integer;
    ef_search::Union{Nothing,Int} = nothing,
) where {T<:LinearAlgebra.BlasFloat}
    size(queries, 1) == index.dimension ||
        throw(DimensionMismatch("Expected queries with $(index.dimension) rows"))

    S = float(T)
    n_queries = size(queries, 2)
    results = Vector{Vector{Neighbor{S}}}(undef, n_queries)
    n_queries == 0 && return results
    if k <= 0 || index.n_points == 0
        @inbounds for i in 1:n_queries
            results[i] = Neighbor{S}[]
        end
        return results
    end

    base_policy = index.traversal_policy
    ef = ef_search === nothing ? default_ef(base_policy) : ef_search
    ef = max(ef, k)

    nworkers = Threads.nthreads()
    # Bounded channel sized to a small multiple of nworkers — workers drain
    # while the producer fills, so a buffer that holds enough work to keep
    # nworkers tasks busy is sufficient. Using n_queries directly here would
    # allocate proportional to the batch size (e.g. 8 MB for a 1M-query
    # batch) for no benefit.
    work = Channel{Int}(min(n_queries, max(64, 4 * nworkers)); spawn=true) do ch
        @inbounds for i in 1:n_queries
            put!(ch, i)
        end
    end

    workers = Vector{Task}(undef, nworkers)
    for w in 1:nworkers
        workers[w] = Threads.@spawn begin
            # Per-task scratch — captured by closure, never indexed by
            # Threads.threadid() (which can change across yield points in
            # Julia 1.10+). Each task owns its scratch for life. The
            # StampVisited generation buffer + the heap-data Vectors are
            # reused across every query this worker processes; that's the
            # whole point of the pool.
            scratch = BatchQueryScratch{S}(index.n_points, ef)
            for i in work
                q = @view queries[:, i]
                visited = _acquire_batch_visited!(scratch, index.n_points)
                results[i] = _query_search(
                    index, data, q, k, ef, visited,
                    scratch.best_data, scratch.pending_data,
                )
            end
        end
    end
    foreach(wait, workers)

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

# Pooled variant of _search_layer. Uses caller-supplied buffers for the
# best-heap and pending-heap backing storage (cleared via empty! so any
# pre-existing capacity is preserved across queries within a worker).
# Otherwise identical to _search_layer.
function _search_layer_pooled(
    index::HNSWIndex,
    layer::Int,
    entry::NeighborCandidate{T},
    q,
    data,
    ef::Int,
    visited,
    best_buf::Vector{NeighborCandidate{T}},
    pend_buf::Vector{NeighborCandidate{T}},
) where {T}
    policy = with_ef(index.traversal_policy, ef)
    empty!(best_buf)
    empty!(pend_buf)
    best_heap = BestCandidatesHeap{T}(best_buf, policy.ef_search)
    push!(best_heap, entry)
    pending = NeighborMinHeap{T}(pend_buf)
    push!(pending, entry)
    state = TraversalState{T}(pending, best_heap)
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
    sort!(state.best.data, alg = Base.Sort.MergeSort, by = c -> c.dist)
    return state.best.data
end

# Used by the BUILD path. The query path uses `_search_layer_pooled` (above)
# instead. If you change semantics here (early-exit gate, sort algorithm,
# visited-mark order, etc.) — change `_search_layer_pooled` to match. Only
# the heap-construction site differs between the two.
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

# --------------------------------------------------------------------------
# Threaded build path
# --------------------------------------------------------------------------
#
# Two-phase build:
#   Phase 1 (serial): sample levels for all nodes, allocate all per-node slots
#                     in every required layer up-front. After this phase no
#                     thread needs to grow `index.layers` or per-layer Vectors.
#   Phase 2 (parallel): each thread inserts a node by running greedy descent
#                       + ef-search + connect, using:
#                         - per-thread StampVisited
#                         - per-thread RNG (already pre-sampled levels, but
#                           the search machinery may need an RNG in future)
#                         - per-node ReentrantLock to serialise adjacency
#                           reads (briefly, copy to scratch) and writes
#                         - global lock for entry_point/max_layer mutations
#
# Determinism: NOT bitwise-deterministic — thread interleaving determines the
# order in which insertions complete, and a node inserted later sees a
# slightly different graph than under serial insertion. Recall is preserved
# in expectation (same contract as PyNNDescent's n_jobs>1 and hnswlib).

# Lock-aware neighbor read: copy adjacency[id] into thread-local scratch under
# the node's lock, return scratch. For threaded path only.
@inline function _read_neighbors_threaded!(
    scratch::Vector{Int},
    adjacency::HNSWLayer,
    id::Int,
    lock::ReentrantLock,
)
    Base.lock(lock)
    try
        src = adjacency[id]
        len = length(src)
        resize!(scratch, len)
        @inbounds for i in 1:len
            scratch[i] = src[i]
        end
    finally
        Base.unlock(lock)
    end
    return scratch
end

# Threaded variant of _greedy_descent: reads `adjacency[current]` under lock
# into a thread-local scratch, then iterates the snapshot.
function _greedy_descent_threaded(
    index::HNSWIndex,
    layer::Int,
    entry_id::Int,
    q,
    data,
    nbr_scratch::Vector{Int},
)
    current = entry_id
    current_dist = index.distance(@view(data[:, current]), q)
    adjacency = index.layers[layer + 1]
    locks = index.node_locks
    improved = true
    while improved
        improved = false
        nbrs = _read_neighbors_threaded!(nbr_scratch, adjacency, current, locks[current])
        @inbounds for neighbor in nbrs
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

# Threaded variant of _search_layer: reads each node's neighbor list into
# thread-local scratch under that node's lock before iterating.
function _search_layer_threaded(
    index::HNSWIndex,
    layer::Int,
    entry::NeighborCandidate{T},
    q,
    data,
    ef::Int,
    visited::StampVisited,
    nbr_scratch::Vector{Int},
) where {T}
    policy = with_ef(index.traversal_policy, ef)
    state = initialize_state(policy, entry)
    _mark_visited!(visited, entry.id)
    adjacency = index.layers[layer + 1]
    locks = index.node_locks

    while should_continue(policy, state)
        current = pop_pending!(policy, state)
        worst = worst_distance(policy, state)
        if current.dist > worst
            break
        end
        nbrs = _read_neighbors_threaded!(nbr_scratch, adjacency, current.id, locks[current.id])
        @inbounds for neighbor in nbrs
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
    sort!(state.best.data, alg = Base.Sort.MergeSort, by = c -> c.dist)
    return state.best.data
end

# Threaded variant of _connect_new_node!:
#   * Lock node_id's slot, write its own list, unlock.
#   * For each neighbor: lock, push reverse edge, prune, unlock.
# Locks are acquired one at a time (no two simultaneously) so deadlock is
# impossible.
function _connect_new_node_threaded!(
    index::HNSWIndex,
    layer::Int,
    node_id::Int,
    neighbors::Vector{NeighborCandidate{T}},
    data,
) where {T}
    adjacency = index.layers[layer + 1]
    locks = index.node_locks

    # Write the new node's adjacency under its own lock.
    own_lock = locks[node_id]
    Base.lock(own_lock)
    try
        own_list = adjacency[node_id]
        resize!(own_list, length(neighbors))
        @inbounds for (i, n) in enumerate(neighbors)
            own_list[i] = n.id
        end
    finally
        Base.unlock(own_lock)
    end

    # Update reverse edges. Each neighbor's lock acquired and released
    # independently — no two locks held simultaneously.
    for neighbor in neighbors
        nlock = locks[neighbor.id]
        Base.lock(nlock)
        try
            list_b = adjacency[neighbor.id]
            push!(list_b, node_id)
            _prune_list!(index, list_b, neighbor.id, data, max_degree(index.neighbor_policy))
        finally
            Base.unlock(nlock)
        end
    end
    return
end

# Two-phase threaded build.
function _build_index_threaded!(
    index::HNSWIndex,
    data::AbstractMatrix,
    n::Int,
    rng::AbstractRNG,
)
    # Phase 1: pre-sample levels.
    levels = Vector{Int}(undef, n)
    for i in 1:n
        levels[i] = sample_layer(index.planner, rng)
    end
    max_level = maximum(levels)

    # Allocate all layers up-front, with all n slots populated. After this
    # block no thread mutates `index.layers` or grows per-layer Vectors.
    cap = max_degree(index.neighbor_policy) + 1
    resize!(index.layers, max_level + 1)
    for li in 1:(max_level + 1)
        layer = Vector{NeighborList}(undef, n)
        @inbounds for i in 1:n
            v = Int[]
            sizehint!(v, cap)
            layer[i] = v
        end
        index.layers[li] = layer
    end

    # Per-node locks.
    resize!(index.node_locks, n)
    @inbounds for i in 1:n
        index.node_locks[i] = ReentrantLock()
    end

    # Insert node 1 serially (entry-point bootstrap).
    index.entry_point = 1
    index.max_layer = levels[1]
    index.n_points = 1

    # Worker-pool pattern: spawn `nthreads` long-lived tasks; each owns its
    # own visit buffer + neighbor scratch. Tasks pull work from a Channel, so
    # a task that gets migrated to a different OS thread still uses its OWN
    # buffer (closure captures by reference, not by Threads.threadid()).
    # This is the only safe pattern under Julia 1.10+ where ReentrantLock
    # acquisition is a yield point and `threadid()` is unstable across it.
    nthreads = Threads.nthreads()
    work = Channel{Int}(max(n, 1))
    for nid in 2:n
        put!(work, nid)
    end
    close(work)

    workers = Vector{Task}(undef, nthreads)
    for t in 1:nthreads
        workers[t] = Threads.@spawn begin
            # Per-task state — captured by closure, NOT indexed by threadid.
            visit_buf = zeros(UInt32, n)
            visit_gen = UInt32(0)
            nbr_scratch = Int[]
            sizehint!(nbr_scratch, cap)

            for node_id in work
                level = levels[node_id]
                point = @view data[:, node_id]

                # Snapshot entry/max under global lock. The values may be
                # stale by the time we use them; that's tolerated (recall
                # may suffer slightly, but no structural defect — see
                # connect-loop comment below).
                Base.lock(index.global_lock)
                cur_entry = index.entry_point
                cur_max = index.max_layer
                Base.unlock(index.global_lock)

                # Greedy descent on layers above this node's level.
                current = cur_entry
                current_dist = index.distance(@view(data[:, current]), point)
                for layer = cur_max:-1:max(level + 1, 1)
                    cand = _greedy_descent_threaded(index, layer, current, point, data, nbr_scratch)
                    current = cand.id
                    current_dist = cand.dist
                end

                # Connect loop on layers min(level, cur_max)..0.
                # If level > cur_max, layers cur_max+1..level are intentionally
                # skipped here — same as serial HNSW when this node becomes
                # the new highest-level node alone. Other nodes at those
                # layers (added concurrently by other threads) won't have a
                # reverse edge to this node, but the global-lock update at
                # the end ensures this node becomes entry_point if level is
                # still the max. The race where another thread overtakes us
                # past `level` mid-execution is handled by the conditional
                # update — we just don't become entry_point in that case.
                for layer = min(level, cur_max):-1:0
                    visit_gen += UInt32(1)
                    if visit_gen == UInt32(0)
                        fill!(visit_buf, UInt32(0))
                        visit_gen = UInt32(1)
                    end
                    visited = StampVisited(visit_buf, visit_gen)

                    results = _search_layer_threaded(
                        index, layer,
                        NeighborCandidate(current, current_dist),
                        point, data, index.ef_construction, visited, nbr_scratch,
                    )
                    neighbors = select_neighbors(index.neighbor_policy, results, data, index.distance)
                    _connect_new_node_threaded!(index, layer, node_id, neighbors, data)
                    if !isempty(neighbors)
                        current = neighbors[1].id
                        current_dist = neighbors[1].dist
                    end
                end

                # Update entry point if we exceeded the current max_layer.
                if level > cur_max
                    Base.lock(index.global_lock)
                    try
                        if level > index.max_layer
                            index.max_layer = level
                            index.entry_point = node_id
                        end
                    finally
                        Base.unlock(index.global_lock)
                    end
                end
            end
        end
    end
    foreach(wait, workers)

    index.n_points = n
    return index
end
