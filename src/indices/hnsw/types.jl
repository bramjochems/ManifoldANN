using LinearAlgebra
using Random

# INVARIANT: a NeighborList contains unique ids. The sole writer is
# `_connect_new_node!` (src/indices/hnsw/query.jl), which pushes each new
# `node_id` exactly once into a neighbor's list. `_prune_list!` relies on
# this invariant — there is no `unique!` defense. Any future code path that
# mutates adjacency (e.g. graph import, `add_edge!`-style API) MUST preserve
# uniqueness, or `_prune_list!`'s diversified scan will misbehave.
const NeighborList = Vector{Int}

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

Adjacency representation for a single HNSW layer. Each node owns a mutable
vector of neighbor ids.
"""
const HNSWLayer = Vector{NeighborList}

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
- `D`: Distance function type (must be thread-safe)
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
    # zeroed (UInt32 is the deliberate width — 2^32 calls amortizes the wrap
    # cost; widening to UInt64 makes the wrap branch effectively unreachable
    # and a buffer-zeroing bug becomes a 4-billion-call invariant).
    #
    # COUPLING: this buffer is index-state, used ONLY by the single-threaded
    # build path via `_acquire_build_visited!`. Concurrent build uses
    # per-thread StampVisited instead (see `_build_index_threaded!`).
    visit_stamps::Vector{UInt32}
    visit_generation::UInt32
    # Per-node lock used by the threaded build path. `length(node_locks) == 0`
    # signals the single-threaded path (no locking overhead). When threaded,
    # locks are acquired during adjacency reads (briefly, to copy into a
    # thread-local scratch) and writes (during _connect_new_node!).
    node_locks::Vector{ReentrantLock}
    # Global lock for entry_point / max_layer / layers expansion in threaded
    # builds. Rare contention (~log n events). Held only at the start/end of
    # an insertion when level > current max_layer.
    global_lock::ReentrantLock
end

index_distance(index::HNSWIndex) = index.distance

configured_k(::HNSWIndex) = nothing
supports_mutation(::HNSWIndex) = true
