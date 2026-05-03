using LinearAlgebra
using Random

"""
    NeighborCandidate{T}

Helper struct carrying a node id plus its distance to the current query. The
distance type `T` remains parametric so we avoid widening (e.g., Float32 → Float64)
when the active distance metric already matches the data precision.
"""
struct NeighborCandidate{T<:AbstractFloat}
    id::Int
    dist::T
end

"""
    HNSWLayer

Slab adjacency for one HNSW layer. Column `i` of `neighbors` holds node `i`'s
neighbor ids; only the prefix `1:degree[i]` is live (slots beyond hold stale
ids from earlier prune iterations — never zeros/garbage, so a racing reader
sees valid-but-possibly-stale ids, which is harmless: `_was_visited` already
guards re-entry, and distance recomputation on a stale id is a no-op
correctness-wise).

INVARIANTS:
- `1 <= degree[i] <= capacity` for all live nodes; sole writer is the
  build/insert! path.
- `neighbors[1:degree[i], i]` contains unique ids — `_prune_list!` relies on
  this; any code path mutating adjacency must preserve uniqueness.
- A node never appears in its own list.

Reads are LOCK-FREE: load `degree[id]` (acquire) then iterate
`neighbors[1:deg, id]`. Writes (`_connect_new_node!`'s reverse-edge prune)
take the per-node `SpinLock` in `HNSWIndex.node_locks`.
"""
mutable struct HNSWLayer
    neighbors::Matrix{Int}
    degree::Vector{Int}
    capacity::Int  # = max_degree(neighbor_policy) + 1
end

HNSWLayer(n::Int, capacity::Int) =
    HNSWLayer(zeros(Int, capacity, n), zeros(Int, n), capacity)

@inline node_count(layer::HNSWLayer) = length(layer.degree)

"""
    layer_neighbors(layer, id) -> SubArray{Int}

Return a view over node `id`'s live neighbors in `layer`. Iteration / `length`
/ `sort` work as expected. The view aliases the underlying slab column —
mutating it mutates the graph.
"""
@inline layer_neighbors(layer::HNSWLayer, id::Int) =
    @inbounds view(layer.neighbors, 1:layer.degree[id], id)

"""
    BatchQueryScratch{T}

Per-worker scratch state for the BATCH query path. Held in a per-index pool
(`HNSWIndex.query_scratch_pool`) and reused across all queries any worker
processes, so its allocation cost amortises across the full lifetime of the
index.

Holds:
- a generation-stamped visited buffer (sized to `n_points`)
- backing `Vector{NeighborCandidate{T}}` for the `best` (max-)heap
- backing `Vector{NeighborCandidate{T}}` for the `pending` (min-)heap

Lifetime is owned by one task at a time via the pool's acquire/release.
Concurrent use from multiple threads on the same `BatchQueryScratch` is UB.

NOT used by the single-query API — that path allocates a `BitSetVisited`
plus fresh empty heap buffers per call (small, one-shot).
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

"""
    BatchScratchPool{T}

Lock-protected pool of `BatchQueryScratch{T}` buffers, owned by an
`HNSWIndex`. Acquire returns either a recycled buffer or freshly-allocated
one; release pushes a buffer back, capped at `capacity` (extras are
dropped to bound memory).

Lifetime: per-index. The pool is touched only by the batch `query` path
(which holds the contract that the index is not concurrently mutated, see
HNSWIndex docstring), so there is no interaction with build/insert!.
"""
mutable struct BatchScratchPool{T<:AbstractFloat}
    buffers::Vector{BatchQueryScratch{T}}
    capacity::Int
    lock::Threads.SpinLock
end

BatchScratchPool{T}(capacity::Int = 0) where {T<:AbstractFloat} =
    BatchScratchPool{T}(Vector{BatchQueryScratch{T}}(), capacity, Threads.SpinLock())

# Acquire a scratch buffer: pop one from the pool if available, else
# allocate a fresh one. Stamp buffer is resized to n_points if needed.
function acquire_scratch!(pool::BatchScratchPool{T}, n_points::Int, ef_capacity::Int) where {T}
    scratch = nothing
    Base.lock(pool.lock)
    try
        if !isempty(pool.buffers)
            scratch = pop!(pool.buffers)
        end
    finally
        Base.unlock(pool.lock)
    end
    if scratch === nothing
        return BatchQueryScratch{T}(n_points, ef_capacity)
    end
    # Resize stamps if the index grew between acquires.
    if length(scratch.visit_stamps) < n_points
        old = length(scratch.visit_stamps)
        resize!(scratch.visit_stamps, n_points)
        @inbounds for i in (old+1):n_points
            scratch.visit_stamps[i] = UInt32(0)
        end
    end
    return scratch
end

# Release a scratch buffer back to the pool. Drops the buffer if the pool
# is at capacity (memory bound).
function release_scratch!(pool::BatchScratchPool{T}, scratch::BatchQueryScratch{T}) where {T}
    Base.lock(pool.lock)
    try
        if length(pool.buffers) < pool.capacity
            push!(pool.buffers, scratch)
        end
    finally
        Base.unlock(pool.lock)
    end
    return
end

"""
    HNSWIndex

Hierarchical Navigable Small World graph-based index with pluggable layer
planners, neighbor-selection policies, and traversal strategies. Stores only
the structural graph metadata so callers remain responsible for supplying the
dataset at query time.

# Type Parameters
- `T`: Element type (e.g., Float32, Float64)
- `LP`: Layer planner type
- `NP`: Neighbor policy type
- `TP`: Traversal policy type
- `D`: Distance function type. **Must be re-entrant / thread-safe**: the
  batch `query(index, data, queries::Matrix, k)` and the threaded
  `build_index(...; threaded=true)` paths call this concurrently from
  multiple worker tasks. The default `default_distance` and Distances.jl
  metrics satisfy this. A stateful distance functor (e.g. with internal
  cache) does NOT.
"""
mutable struct HNSWIndex{T<:LinearAlgebra.BlasFloat,LP,NP,TP,D} <: AbstractANNIndex
    layers::Vector{HNSWLayer}
    entry_point::Int
    max_layer::Int
    dimension::Int
    n_points::Int
    M::Int
    ef_construction::Int
    planner::LP
    neighbor_policy::NP
    traversal_policy::TP
    distance::D
    # Generation-stamped visited buffer. Replaces per-call BitSet allocation
    # in _search_layer during build. `visit_stamps[i] == visit_generation` ⟺
    # node `i` has been visited in the current traversal. Bumping
    # `visit_generation` is an O(1) reset. On UInt32 wrap, the buffer is
    # zeroed.
    #
    # COUPLING: this buffer is index-state, used ONLY by the single-threaded
    # build path via `_acquire_build_visited!`. Concurrent build uses
    # per-thread StampVisited instead.
    visit_stamps::Vector{UInt32}
    visit_generation::UInt32
    # Per-node SpinLock used by the threaded build path WRITERS only. Reads
    # are lock-free (atomic-acquire load on the layer's `degree[id]`, then
    # iterate the slab column). `length(node_locks) == 0` signals the
    # single-threaded path.
    node_locks::Vector{Threads.SpinLock}
    # Global lock for entry_point / max_layer / layers expansion in threaded
    # builds. Rare contention (~log n events). Held only at the start/end of
    # an insertion when level > current max_layer.
    global_lock::ReentrantLock
    # Pool of reusable BatchQueryScratch buffers for the batch-query path.
    # Index-lifetime; eliminates the per-call allocation of stamp + heap
    # buffers. Capacity is set at build time to `2 * Threads.nthreads()` so
    # that all workers can concurrently hold a buffer with headroom for
    # pipelined acquires.
    query_scratch_pool::BatchScratchPool{T}
end

index_distance(index::HNSWIndex) = index.distance

configured_k(::HNSWIndex) = nothing
supports_mutation(::HNSWIndex) = true
