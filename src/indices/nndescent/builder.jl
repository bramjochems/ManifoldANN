using Random

"""
    build_index(NNDescentIndex, data; kwargs...)

Construct an NN-Descent index over `data`. Keeps only the neighbor graph, so
callers must still pass `data` when querying. Keywords:

- `k`: Number of neighbors per point (default 32)
- `max_iterations`: Maximum NN-Descent iterations (default 10)
- `convergence_threshold`: Relative improvement threshold to stop early
- `sampling_policy`: `UniformPairSampling`, `:uniform`, or `nothing` (defaults to uniform)
- `symmetry_policy`: `FullSymmetry()`, `PrunedSymmetry(1.5)`, `NoSymmetry()`, or symbol (defaults to `:full`)
- `apply_symmetry_continuously`: If true, apply symmetry after each iteration; if false, apply only at end (default false)
- `rng`: RNG used for initialization and sampling
- `distance`: Distance function (defaults to `default_squared_distance`)
- `pruning_degree_multiplier`: Per-iteration candidate set per node is capped at
  `ceil(pruning_degree_multiplier × k)` (default 1.5, matches PyNNDescent).
  Larger values increase recall at quadratic build cost.
- `max_candidate_neighbors`: Explicit override for the per-iteration candidate
  cap. If `nothing` (default), derived from `pruning_degree_multiplier × k`.
- `threaded`: Run the local-join in parallel using a worker-pool of
  `Threads.nthreads()` long-lived tasks pulling work from a `Channel` (default
  `true`). The threaded path acquires per-node `ReentrantLock`s on heap
  reads and mutations; this preserves graph quality but **gives up bitwise
  determinism**: two builds with the same `rng` may produce different
  neighbor lists because thread interleaving determines insertion order.
  Pass `threaded=false` for reproducible builds (matches PyNNDescent's
  `n_jobs=1` semantics).
- `init`: Initial-graph strategy. `:random` (default) is the original
  bidirectional random init. `:rptree` builds an RP-tree forest and seeds
  each node's heap with the closest k from the union of co-leaf members
  across the forest, also bidirectionally. The RP-tree path was benchmarked
  as slower for build time on this codebase (see commit history) but
  improves recall at moderate n (n=5000 d=32 k=15: 0.92 → 0.96). Opt in
  via `init=:rptree` if recall matters more than build speed.
- `n_trees`, `leaf_cap`: RP-tree forest parameters; only used when
  `init=:rptree`. Defaults derived from n and k (PyNNDescent's heuristics).
"""
function build_index(
    ::Type{NNDescentIndex},
    data::AbstractMatrix{T};
    k::Int = NNDESCENT_DEFAULT_K,
    max_iterations::Int = NNDESCENT_DEFAULT_MAX_ITERATIONS,
    convergence_threshold::Float64 = NNDESCENT_DEFAULT_CONVERGENCE_THRESHOLD,
    sampling_policy::Union{AbstractNNDescentSamplingPolicy,Symbol,Nothing} = nothing,
    symmetry_policy::Union{AbstractSymmetryPolicy,Symbol,Nothing} = nothing,
    apply_symmetry_continuously::Bool = false,
    rng::AbstractRNG = Random.default_rng(),
    distance::D = default_squared_distance,
    pruning_degree_multiplier::Real = NNDESCENT_DEFAULT_PRUNING_DEGREE_MULTIPLIER,
    max_candidate_neighbors::Union{Int,Nothing} = nothing,
    threaded::Bool = true,
    init::Symbol = :random,
    n_trees::Union{Int,Nothing} = nothing,
    leaf_cap::Union{Int,Nothing} = nothing,
) where {T<:LinearAlgebra.BlasFloat,D}
    d, n = size(data)
    d > 0 || throw(ArgumentError("Dataset must have at least one dimension"))
    n > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    k > 0 || throw(ArgumentError("k must be positive"))
    n > 1 || throw(ArgumentError("NN-Descent requires at least two points"))
    k < n || throw(ArgumentError("k must be less than the number of points ($n)"))
    max_iterations >= 1 ||
        throw(ArgumentError("max_iterations must be at least 1"))
    convergence_threshold >= 0 ||
        throw(ArgumentError("convergence_threshold must be non-negative"))
    pruning_degree_multiplier > 0 ||
        throw(ArgumentError("pruning_degree_multiplier must be positive"))

    # Resolve the candidate cap: explicit override wins, otherwise derive from
    # pruning_degree_multiplier × k. Always at least k so a node can sample its
    # full forward neighborhood.
    resolved_cap = max_candidate_neighbors === nothing ?
        max(k, ceil(Int, pruning_degree_multiplier * k)) :
        max_candidate_neighbors
    resolved_cap > 0 ||
        throw(ArgumentError("resolved candidate cap must be positive"))

    resolved_sampling_policy = _resolve_sampling_policy(sampling_policy)
    resolved_symmetry_policy = _resolve_symmetry_policy(symmetry_policy)

    # Determine the distance element type for neighbor bookkeeping
    probe = distance(view(data, :, 1), view(data, :, 1))
    probe isa AbstractFloat ||
        throw(
            ArgumentError(
                "distance function must return an AbstractFloat, got $(typeof(probe))",
            ),
        )
    dist_type = typeof(probe)

    working_graph =
        [NNDescentNeighborNode{dist_type}(k) for _ in 1:n]::Vector{
            NNDescentNeighborNode{dist_type},
        }

    if init === :random
        _initialize_random_neighbors!(working_graph, data, k, distance, rng)
    elseif init === :rptree
        resolved_n_trees = n_trees === nothing ? default_rptree_n_trees(n) : n_trees
        resolved_leaf_cap = leaf_cap === nothing ? default_rptree_leaf_cap(k) : leaf_cap
        resolved_n_trees > 0 || throw(ArgumentError("n_trees must be positive"))
        resolved_leaf_cap > 0 || throw(ArgumentError("leaf_cap must be positive"))
        _initialize_rptree_neighbors!(
            working_graph, data, k, distance, rng,
            resolved_n_trees, resolved_leaf_cap,
        )
    else
        throw(ArgumentError("Unknown init mode :$init. Use :random or :rptree"))
    end
    _run_nndescent!(
        working_graph,
        data,
        distance,
        k,
        resolved_sampling_policy,
        resolved_symmetry_policy,
        apply_symmetry_continuously,
        max_iterations,
        convergence_threshold,
        rng,
        resolved_cap,
        threaded,
    )

    adjacency = _finalize_neighbors(working_graph, k)

    return NNDescentIndex{
        T,
        typeof(distance),
        typeof(resolved_sampling_policy),
        typeof(resolved_symmetry_policy),
    }(
        d,
        n,
        k,
        max_iterations,
        distance,
        resolved_sampling_policy,
        resolved_symmetry_policy,
        adjacency,
    )
end

function _resolve_sampling_policy(
    policy::Union{AbstractNNDescentSamplingPolicy,Symbol,Nothing},
)
    if policy === nothing
        return UniformPairSampling()
    elseif policy isa AbstractNNDescentSamplingPolicy
        return policy
    elseif policy isa Symbol
        if policy === :uniform
            return UniformPairSampling()
        else
            throw(ArgumentError("Unknown NN-Descent sampling policy $policy"))
        end
    else
        throw(ArgumentError("Unsupported sampling policy type $(typeof(policy))"))
    end
end

function _resolve_symmetry_policy(
    policy::Union{AbstractSymmetryPolicy,Symbol,Nothing},
)
    if policy === nothing
        return FullSymmetry()
    elseif policy isa AbstractSymmetryPolicy
        return policy
    elseif policy isa Symbol
        if policy === :full
            return FullSymmetry()
        elseif policy === :pruned
            return PrunedSymmetry()
        elseif policy === :none
            return NoSymmetry()
        else
            throw(ArgumentError("Unknown symmetry policy $policy. Use :full, :pruned, or :none"))
        end
    else
        throw(ArgumentError("Unsupported symmetry policy type $(typeof(policy))"))
    end
end

function _initialize_random_neighbors!(
    graph::Vector{NNDescentNeighborNode{T}},
    data::AbstractMatrix,
    k::Int,
    distance,
    rng::AbstractRNG,
) where {T}
    n = length(graph)
    @inbounds for i in 1:n
        node = graph[i]
        chosen = Set{Int}()
        added = 0
        while added < k
            candidate = rand(rng, 1:n)
            candidate == i && continue
            candidate in chosen && continue
            push!(chosen, candidate)
            dist = distance(view(data, :, i), view(data, :, candidate))
            # All initial neighbors are "new"
            # Add bidirectional edges (like NearestNeighborDescent.jl)
            # This improves initial graph connectivity significantly
            push!(node.new_neighbors, candidate, dist)
            # Add reverse edge using unsafe_push! to allow exceeding capacity
            # Symmetry policy will prune back to appropriate size later
            unsafe_push!(graph[candidate].new_neighbors, i, dist)
            added += 1
        end
    end
    return nothing
end

# Serial local-join for one NN-Descent iteration. Returns the number of
# successful neighbor updates this iteration.
function _local_join_serial!(
    graph::Vector{NNDescentNeighborNode{T}}, data, distance,
    sampling_policy, max_candidate_neighbors,
    r_new, r_old, new_scratch::Vector{Int}, old_scratch::Vector{Int},
    rng::AbstractRNG,
) where {T}
    updates = 0
    @inbounds for node_idx in 1:length(graph)
        node = graph[node_idx]
        (isempty(node.new_neighbors) && isempty(r_new[node_idx])) && continue
        _sample_neighbor_ids_with_reverse!(
            new_scratch, node.new_neighbors, r_new[node_idx],
            max_candidate_neighbors, rng)
        isempty(new_scratch) && continue
        new_candidates = new_scratch
        _sample_neighbor_ids_with_reverse!(
            old_scratch, node.old_neighbors, r_old[node_idx],
            max_candidate_neighbors, rng)
        old_candidates = old_scratch

        for new_idx in 1:length(new_candidates)
            src = new_candidates[new_idx]
            for inner_idx in (new_idx + 1):length(new_candidates)
                dst = new_candidates[inner_idx]
                should_consider_pair(sampling_policy, rng) || continue
                updates += _connect_pair!(graph, data, distance, src, dst)
            end
            for dst in old_candidates
                src == dst && continue
                should_consider_pair(sampling_policy, rng) || continue
                updates += _connect_pair!(graph, data, distance, src, dst)
            end
        end
    end
    return updates
end

# Threaded local-join for one NN-Descent iteration.
#
# Concurrency model:
#   * Each task owns its sampling scratch buffers and RNG via closure capture
#     (NOT Threads.threadid() indexing). `lock(::ReentrantLock)` is a yield
#     point in Julia 1.10+, the scheduler may resume a task on a different OS
#     thread, but threadid is captured stale — so threadid-indexed buffers are
#     unsafe under contention. Worker-pool with closure-captured per-task state
#     is the canonical safe pattern.
#   * Reads of `graph[node_idx].{new,old}_neighbors` are done under
#     `locks[node_idx]` by snapshotting the heap's ids into a per-task scratch
#     under the lock, then sampling/iterating the snapshot lock-free. This
#     removes the read-during-mutate race against `_connect_pair_locked!` /
#     `_insert_neighbor!` running on another task — the writer can be
#     `push!`-ing and `_heap_sift_up!`-ing the same heap concurrently, which
#     reallocates the underlying Vector and produces torn reads / iterator
#     invalidation. (Mirrors `_read_neighbors_threaded!` in HNSW.)
function _local_join_threaded!(
    graph::Vector{NNDescentNeighborNode{T}}, data, distance,
    sampling_policy, max_candidate_neighbors,
    r_new, r_old, locks, parent_rng::AbstractRNG,
) where {T}
    n = length(graph)
    updates_atomic = Threads.Atomic{Int}(0)
    nthreads = Threads.nthreads()

    # Distribute node ids into a Channel; each worker pulls work as it goes.
    work = Channel{Int}(max(n, 1))
    for nid in 1:n
        put!(work, nid)
    end
    close(work)

    # Pre-derive per-task RNG seeds from the parent RNG so streams diverge but
    # remain reproducible up to thread interleaving (the threaded path is not
    # bitwise-deterministic by design — see build_index docstring).
    task_seeds = [rand(parent_rng, UInt64) for _ in 1:nthreads]

    workers = Vector{Task}(undef, nthreads)
    for t in 1:nthreads
        seed = task_seeds[t]
        workers[t] = Threads.@spawn begin
            # Per-task state — captured by closure, NOT indexed by threadid.
            new_scratch = Int[]; sizehint!(new_scratch, max_candidate_neighbors)
            old_scratch = Int[]; sizehint!(old_scratch, max_candidate_neighbors)
            new_snap = Int[]; sizehint!(new_snap, max_candidate_neighbors)
            old_snap = Int[]; sizehint!(old_snap, max_candidate_neighbors)
            trng = Random.Xoshiro(seed)
            local_updates = 0

            for node_idx in work
                @inbounds begin
                    node = graph[node_idx]
                    # Snapshot heap ids under the node's lock; release before
                    # sampling/iterating. Brief crit-section, race-free.
                    Base.lock(locks[node_idx])
                    try
                        empty!(new_snap)
                        for nb in node.new_neighbors
                            push!(new_snap, nb.id)
                        end
                        empty!(old_snap)
                        for nb in node.old_neighbors
                            push!(old_snap, nb.id)
                        end
                    finally
                        Base.unlock(locks[node_idx])
                    end

                    (isempty(new_snap) && isempty(r_new[node_idx])) && continue
                    _sample_ids_from_vector_with_reverse!(
                        new_scratch, new_snap, r_new[node_idx],
                        max_candidate_neighbors, trng)
                    isempty(new_scratch) && continue
                    new_candidates = new_scratch
                    _sample_ids_from_vector_with_reverse!(
                        old_scratch, old_snap, r_old[node_idx],
                        max_candidate_neighbors, trng)
                    old_candidates = old_scratch

                    for new_idx in 1:length(new_candidates)
                        src = new_candidates[new_idx]
                        for inner_idx in (new_idx + 1):length(new_candidates)
                            dst = new_candidates[inner_idx]
                            should_consider_pair(sampling_policy, trng) || continue
                            local_updates += _connect_pair_locked!(
                                graph, data, distance, src, dst, locks)
                        end
                        for dst in old_candidates
                            src == dst && continue
                            should_consider_pair(sampling_policy, trng) || continue
                            local_updates += _connect_pair_locked!(
                                graph, data, distance, src, dst, locks)
                        end
                    end
                end
            end
            Threads.atomic_add!(updates_atomic, local_updates)
        end
    end
    foreach(wait, workers)
    return updates_atomic[]
end

function _run_nndescent!(
    graph::Vector{NNDescentNeighborNode{T}},
    data::AbstractMatrix,
    distance,
    k::Int,
    sampling_policy::AbstractNNDescentSamplingPolicy,
    symmetry_policy::AbstractSymmetryPolicy,
    apply_symmetry_continuously::Bool,
    max_iterations::Int,
    convergence_threshold::Float64,
    rng::AbstractRNG,
    max_candidate_neighbors::Int,
    threaded::Bool,
) where {T}
    n = length(graph)
    total_edges = n * k  # Total edges in graph (used to compute improvement ratio)
    # Reverse-neighbor buffers reused across iterations; refilled in place from
    # the current forward graph at the top of each iteration. Hint capacity at
    # 2k: expected steady-state reverse degree equals forward degree k, doubled
    # to absorb skew (popular nodes accumulate more reverse edges).
    r_new, r_old = _allocate_reverse_buffers(n, 2k)

    if threaded
        # Per-task state lives inside `_local_join_threaded!` (closure-captured
        # per worker task, not threadid-indexed — Julia 1.10+ task migration
        # makes threadid unstable across yield points). We only need the
        # per-node lock vector at this scope.
        locks = [ReentrantLock() for _ in 1:n]
    else
        # Serial path needs only one set of scratch buffers and the parent RNG.
        new_scratch = Int[]; sizehint!(new_scratch, max_candidate_neighbors)
        old_scratch = Int[]; sizehint!(old_scratch, max_candidate_neighbors)
    end

    for iteration in 1:max_iterations
        # Refill transient reverse-neighbor lists for this iteration. The
        # canonical local-join uses B[v] ∪ sample(R[v]) for both new and old
        # forward sets; without R, descent is forward-only and converges to
        # lower graph quality (Dong, Charikar, Li 2011).
        _refill_reverse_neighbors!(r_new, r_old, graph)

        updates = if threaded
            _local_join_threaded!(
                graph, data, distance, sampling_policy, max_candidate_neighbors,
                r_new, r_old, locks, rng)
        else
            _local_join_serial!(
                graph, data, distance, sampling_policy, max_candidate_neighbors,
                r_new, r_old, new_scratch, old_scratch, rng)
        end

        # Apply symmetry policy after each iteration if requested
        # This maintains graph symmetry during construction, improving neighbor discovery
        if apply_symmetry_continuously
            apply_symmetry_policy!(graph, k, symmetry_policy)
        end

        # Move all "new" neighbors to "old" for next iteration
        @inbounds for node_idx in 1:n
            _transition_neighbors!(graph[node_idx])
        end

        # Compute relative improvement as fraction of total possible edges updated
        improvement = total_edges == 0 ? 0.0 : updates / total_edges
        if improvement < convergence_threshold
            break
        end
    end

    # Apply symmetry policy at the end if not applied continuously
    # When applied only once at the end, this is more efficient but may reduce graph quality
    if !apply_symmetry_continuously
        apply_symmetry_policy!(graph, k, symmetry_policy)
    end

    return nothing
end

"""
    _transition_neighbors!(node)

Move all "new" neighbors to the "old" heap for the next iteration. This clears
the new_neighbors heap, preparing it for the next round of discoveries.
"""
function _transition_neighbors!(node::NNDescentNeighborNode{T}) where {T}
    # Transfer all new neighbors to old neighbors
    for neighbor in node.new_neighbors
        push!(node.old_neighbors, neighbor.id, neighbor.dist)
    end
    # Clear the new neighbors heap for next iteration
    empty!(node.new_neighbors.data)
    return nothing
end

"""
    apply_symmetry_policy!(graph, k, policy)

Apply the specified symmetry policy to the graph after NN-Descent iterations complete.
"""
function apply_symmetry_policy!(
    graph::Vector{NNDescentNeighborNode{T}},
    k::Int,
    policy::AbstractSymmetryPolicy,
) where {T}
    # Dispatch to the appropriate implementation
    _apply_symmetry!(graph, k, policy)
end

"""
    _apply_symmetry!(graph, k, ::NoSymmetry)

No-op: keep the directed k-NN graph as-is.
"""
function _apply_symmetry!(
    graph::Vector{NNDescentNeighborNode{T}},
    k::Int,
    ::NoSymmetry,
) where {T}
    # Nothing to do - graph remains asymmetric
    return nothing
end

"""
    _apply_symmetry!(graph, k, ::FullSymmetry)

Add all reverse edges to ensure complete graph symmetry.
Nodes may end up with > k neighbors.
"""
function _apply_symmetry!(
    graph::Vector{NNDescentNeighborNode{T}},
    k::Int,
    ::FullSymmetry,
) where {T}
    n = length(graph)

    # Build sets for O(1) membership checking - combine both heaps
    neighbor_sets = [Set{Int}() for _ in 1:n]
    @inbounds for i in 1:n
        for nb in graph[i].old_neighbors
            push!(neighbor_sets[i], nb.id)
        end
        for nb in graph[i].new_neighbors
            push!(neighbor_sets[i], nb.id)
        end
    end

    # Collect reverse edges to add
    to_add = [Vector{Tuple{Int,T}}() for _ in 1:n]

    @inbounds for i in 1:n
        # Check old neighbors
        for nb in graph[i].old_neighbors
            if i ∉ neighbor_sets[nb.id]
                push!(to_add[nb.id], (i, nb.dist))
            end
        end
        # Check new neighbors
        for nb in graph[i].new_neighbors
            if i ∉ neighbor_sets[nb.id]
                push!(to_add[nb.id], (i, nb.dist))
            end
        end
    end

    # Add all the reverse edges to old_neighbors (they're not "new" discoveries)
    # Use unsafe_push! to allow exceeding capacity for full symmetry
    @inbounds for i in 1:n
        for (neighbor_id, dist) in to_add[i]
            # Note: This may exceed k capacity, which is expected for full symmetry
            unsafe_push!(graph[i].old_neighbors, neighbor_id, dist)
        end
    end

    return nothing
end

"""
    _apply_symmetry!(graph, k, policy::PrunedSymmetry)

Add reverse edges like FullSymmetry, but prune each node's combined neighbor list
to at most `policy.degree_multiplier * k` edges. This balances search quality
with memory efficiency, following PyNNDescent's approach.
"""
function _apply_symmetry!(
    graph::Vector{NNDescentNeighborNode{T}},
    k::Int,
    policy::PrunedSymmetry,
) where {T}
    n = length(graph)
    max_degree = ceil(Int, policy.degree_multiplier * k)

    # Build sets for O(1) membership checking - combine both heaps
    neighbor_sets = [Set{Int}() for _ in 1:n]
    @inbounds for i in 1:n
        for nb in graph[i].old_neighbors
            push!(neighbor_sets[i], nb.id)
        end
        for nb in graph[i].new_neighbors
            push!(neighbor_sets[i], nb.id)
        end
    end

    # Collect reverse edges to add
    to_add = [Vector{Tuple{Int,T}}() for _ in 1:n]

    @inbounds for i in 1:n
        # Check old neighbors
        for nb in graph[i].old_neighbors
            if i ∉ neighbor_sets[nb.id]
                push!(to_add[nb.id], (i, nb.dist))
            end
        end
        # Check new neighbors
        for nb in graph[i].new_neighbors
            if i ∉ neighbor_sets[nb.id]
                push!(to_add[nb.id], (i, nb.dist))
            end
        end
    end

    # Add reverse edges and prune to max_degree
    @inbounds for i in 1:n
        # Collect all neighbors from both heaps plus reverse edges
        all_neighbors = Vector{Neighbor{T}}()
        for nb in graph[i].old_neighbors
            push!(all_neighbors, nb)
        end
        for nb in graph[i].new_neighbors
            push!(all_neighbors, nb)
        end
        for (neighbor_id, dist) in to_add[i]
            push!(all_neighbors, Neighbor{T}(neighbor_id, dist))
        end

        # Sort by distance and keep only the closest max_degree neighbors
        sort!(all_neighbors, by = nb -> nb.dist)
        if length(all_neighbors) > max_degree
            resize!(all_neighbors, max_degree)
        end

        # Rebuild the old_neighbors heap with pruned neighbors, clear new_neighbors
        # NOTE: We directly manipulate heap.data to bypass capacity limits.
        # After empty!(), the heap can accept all_neighbors (up to max_degree)
        # even though max_degree > k (the original heap capacity).
        # This is intentional for PrunedSymmetry where degree_multiplier > 1.0.
        empty!(graph[i].old_neighbors.data)
        empty!(graph[i].new_neighbors.data)
        for nb in all_neighbors
            push!(graph[i].old_neighbors, nb.id, nb.dist)
        end
    end

    return nothing
end

function _connect_pair!(
    graph::Vector{NNDescentNeighborNode{T}},
    data::AbstractMatrix,
    distance,
    a::Int,
    b::Int,
) where {T}
    a == b && return 0
    dist = distance(view(data, :, a), view(data, :, b))
    inserted = 0
    inserted += _insert_neighbor!(graph[a], b, dist)
    inserted += _insert_neighbor!(graph[b], a, dist)
    return inserted
end

# Parallel variant: per-node ReentrantLocks protect mutations of
# graph[a].new_neighbors and graph[b].new_neighbors. Acquired one at a time
# (not nested), so no deadlock-avoidance ordering needed.
function _connect_pair_locked!(
    graph::Vector{NNDescentNeighborNode{T}},
    data::AbstractMatrix,
    distance,
    a::Int,
    b::Int,
    locks::Vector{ReentrantLock},
) where {T}
    a == b && return 0
    dist = distance(view(data, :, a), view(data, :, b))
    inserted = 0
    lock(locks[a])
    try
        inserted += _insert_neighbor!(graph[a], b, dist)
    finally
        unlock(locks[a])
    end
    lock(locks[b])
    try
        inserted += _insert_neighbor!(graph[b], a, dist)
    finally
        unlock(locks[b])
    end
    return inserted
end

function _insert_neighbor!(
    node::NNDescentNeighborNode{T},
    neighbor_id::Int,
    dist::T,
) where {T}
    # Cheap O(k) duplicate check on new_neighbors only. Skips the (existing,
    # better-or-equal) cases that would otherwise burn heap pushes on duplicates
    # and risk evicting good unique entries before _finalize_neighbors. We do
    # not scan old_neighbors: in the local-join path inserts target
    # new_neighbors, and the heap-push itself absorbs any rare cross-heap
    # duplicate at finalize time. Skips the catastrophic O(k log k) update path
    # the previous implementation paid on every duplicate.
    @inbounds for existing in node.new_neighbors
        if existing.id == neighbor_id
            # Already present: never push again. Distance updates for an
            # existing id are intentionally dropped; PyNNDescent does the same,
            # and our quality tests (recall, MRR) confirm this is fine.
            return 0
        end
    end
    accepted = push!(node.new_neighbors, neighbor_id, dist)
    return accepted ? 1 : 0
end

function _finalize_neighbors(
    graph::Vector{NNDescentNeighborNode{T}},
    k::Int,
) where {T}
    adjacency = Vector{Vector{Int}}(undef, length(graph))
    @inbounds for i in eachindex(graph)
        node = graph[i]

        # Collect all neighbors from both heaps without dedup yet — we sort
        # by distance first so the dedup pass keeps the smaller-distance
        # entry per id. (Iterating one heap then the other and dropping later
        # duplicates would arbitrarily prefer whichever heap we visited first
        # regardless of distance, which can drop the shorter-distance edge.)
        all_neighbors = Vector{Neighbor{T}}()
        for nb in node.old_neighbors
            push!(all_neighbors, nb)
        end
        for nb in node.new_neighbors
            push!(all_neighbors, nb)
        end

        # Sort by distance, then dedup keeping the first (smallest-distance)
        # occurrence per id.
        sort!(all_neighbors, by = nb -> nb.dist)
        seen_ids = Set{Int}()
        write = 0
        for nb in all_neighbors
            if !(nb.id in seen_ids)
                push!(seen_ids, nb.id)
                write += 1
                all_neighbors[write] = nb
            end
        end
        resize!(all_neighbors, write)

        # Extract IDs
        ids = Vector{Int}(undef, length(all_neighbors))
        for (j, nb) in enumerate(all_neighbors)
            ids[j] = nb.id
        end

        adjacency[i] = ids
    end
    return adjacency
end

function _sample_neighbor_ids(
    heap::BoundedMaxHeap{T},
    limit::Int,
    rng::AbstractRNG,
) where {T}
    limit <= 0 && return Int[]
    collected = Int[]
    @inbounds for nb in heap
        push!(collected, nb.id)
    end
    len = length(collected)
    len <= limit && return collected
    shuffle!(rng, collected)
    resize!(collected, limit)
    return collected
end

# Merged sample of forward neighbors (heap) and reverse neighbors (id vector).
# Implements `B[v] ∪ sample(R[v])` from Dong, Charikar, Li 2011: the canonical
# local-join sample includes both forward and reverse edges. Without reverse
# edges, descent is forward-only and converges to lower-quality graphs.
# Dedup via sort+unique on the merged vector (cheap at typical merged size of
# 15-50 ids; avoids allocating a Set{Int} per call in the hot path). The
# `scratch` buffer is reused across calls within an iteration.
function _sample_neighbor_ids_with_reverse!(
    scratch::Vector{Int},
    heap::BoundedMaxHeap{T},
    reverse_ids::Vector{Int},
    limit::Int,
    rng::AbstractRNG,
) where {T}
    empty!(scratch)
    limit <= 0 && return scratch
    @inbounds for nb in heap
        push!(scratch, nb.id)
    end
    append!(scratch, reverse_ids)
    isempty(scratch) && return scratch
    sort!(scratch)
    unique!(scratch)
    len = length(scratch)
    len <= limit && return scratch
    shuffle!(rng, scratch)
    resize!(scratch, limit)
    return scratch
end

# Variant of `_sample_neighbor_ids_with_reverse!` that takes a pre-extracted
# id vector instead of a heap. Used by the threaded local-join after the
# heap's ids have been snapshotted under lock — we cannot iterate the live
# heap while another task may be mutating it.
function _sample_ids_from_vector_with_reverse!(
    scratch::Vector{Int},
    forward_ids::Vector{Int},
    reverse_ids::Vector{Int},
    limit::Int,
    rng::AbstractRNG,
)
    empty!(scratch)
    limit <= 0 && return scratch
    append!(scratch, forward_ids)
    append!(scratch, reverse_ids)
    isempty(scratch) && return scratch
    sort!(scratch)
    unique!(scratch)
    len = length(scratch)
    len <= limit && return scratch
    shuffle!(rng, scratch)
    resize!(scratch, limit)
    return scratch
end

# Allocate reverse-neighbor buffers once for the full descent. Each iteration
# clears and refills these vectors, avoiding O(n) Vector{Int} allocations per
# iteration.
function _allocate_reverse_buffers(n::Int, hint::Int = 0)
    r_new = [Int[] for _ in 1:n]
    r_old = [Int[] for _ in 1:n]
    if hint > 0
        for v in 1:n
            sizehint!(r_new[v], hint)
            sizehint!(r_old[v], hint)
        end
    end
    return r_new, r_old
end

# Refill reverse-neighbor lists from the current forward graph. R_new[v] =
# nodes u such that v ∈ new_neighbors(u); R_old similarly. Linear pass over
# the graph; called once per iteration after the forward graph changes.
function _refill_reverse_neighbors!(
    r_new::Vector{Vector{Int}},
    r_old::Vector{Vector{Int}},
    graph::Vector{NNDescentNeighborNode{T}},
) where {T}
    @inbounds for v in eachindex(r_new)
        empty!(r_new[v])
        empty!(r_old[v])
    end
    @inbounds for u in eachindex(graph)
        for nb in graph[u].new_neighbors
            push!(r_new[nb.id], u)
        end
        for nb in graph[u].old_neighbors
            push!(r_old[nb.id], u)
        end
    end
    return nothing
end
