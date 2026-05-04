using Random

"""
    RPTreeForestIndex{T,D}

A forest of random-projection trees wrapped as an `AbstractANNIndex`. At
query time, the leaf bucket each tree routes the query to is union'd into a
single candidate set, which is then brute-force scanned with `index.distance`
to produce the final top-k via a bounded heap. Recall scales with `n_trees`;
trees are independent, so the build is embarrassingly parallel — same
threading pattern as the LSH multi-table build.
"""
mutable struct RPTreeForestIndex{T<:AbstractFloat,D} <: AbstractANNIndex
    trees::Vector{RPTree{T}}
    dimension::Int
    n_points::Int
    leaf_cap::Int
    distance::D
end

index_distance(index::RPTreeForestIndex) = index.distance
configured_k(::RPTreeForestIndex) = nothing
supports_mutation(::RPTreeForestIndex) = false

const RPTREE_FOREST_DEFAULT_N_TREES = 8
const RPTREE_FOREST_DEFAULT_LEAF_CAP = 32

"""
    build_index(RPTreeForestIndex, data; n_trees, leaf_cap, distance, splitter, rng)

Build an `RPTreeForestIndex` of `n_trees` independent random-projection trees.

Each tree is built on its own `Threads.@spawn`-ed task with an RNG derived
from `rng` via `spawn_child_rngs`. Per-tree seeds are computed serially up
front, so the resulting forest is deterministic regardless of thread count
or scheduling order.
"""
function build_index(
    ::Type{RPTreeForestIndex},
    data::AbstractMatrix{T};
    n_trees::Integer = RPTREE_FOREST_DEFAULT_N_TREES,
    leaf_cap::Integer = RPTREE_FOREST_DEFAULT_LEAF_CAP,
    distance::D = default_distance,
    splitter::AbstractRPSplitter = TwoPointSplitter(),
    rng::AbstractRNG = Random.default_rng(),
) where {T<:AbstractFloat,D}
    d, n = size(data)
    d > 0 || throw(ArgumentError("Dataset must have positive dimension"))
    n > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    n_trees > 0 || throw(ArgumentError("n_trees must be positive"))
    leaf_cap >= 1 || throw(ArgumentError("leaf_cap must be >= 1"))

    tree_rngs = spawn_child_rngs(rng, n_trees)
    trees = Vector{RPTree{T}}(undef, n_trees)

    # Each tree is independent (own RNG, own output slot). Same pattern as
    # LSHIndex's per-table build (cd1af49). Seeds are derived serially above
    # so determinism is independent of thread scheduling.
    Threads.@threads for idx in 1:n_trees
        trees[idx] = build_rptree(
            data, Int(leaf_cap); splitter = splitter, rng = tree_rngs[idx])
    end

    return RPTreeForestIndex{T,D}(trees, d, n, Int(leaf_cap), distance)
end

"""
    query(index::RPTreeForestIndex, data, q, k)

Approximate kNN by routing `q` through every tree, taking the union of leaf
buckets as the candidate set, and brute-force scoring with `index.distance`.
Returns up to `k` neighbours sorted by distance. Concurrent-safe: reads only
from the index and uses a per-call scratch BitSet plus heap.
"""
function query(
    index::RPTreeForestIndex{T,D},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer,
) where {T<:AbstractFloat,D}
    validate_index_dimensions(index, data, q)
    S = float(T)
    k <= 0 && return Neighbor{S}[]
    actual_k = min(Int(k), index.n_points)
    actual_k == 0 && return Neighbor{S}[]

    seen = BitSet()
    heap = BoundedMaxHeap{S}(actual_k)
    @inbounds for tree in index.trees
        members = leaf_members(tree, q)
        for j in eachindex(members)
            point_id = members[j]
            (point_id in seen) && continue
            push!(seen, point_id)
            dist = S(index.distance(@view(data[:, point_id]), q))
            push!(heap, point_id, dist)
        end
    end
    return to_sorted_vector(heap)
end
