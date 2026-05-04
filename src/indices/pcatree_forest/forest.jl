using Random

"""
    PCATreeForestIndex{T,D,Sp}

A forest of PCA trees wrapped as an `AbstractANNIndex`. At query time,
the leaf bucket each tree routes the query to is union'd into a single
candidate set, which is then brute-force scanned with `index.distance`
to produce the final top-k via a bounded heap. Recall scales with
`n_trees`; trees are independent, so the build is embarrassingly
parallel — same threading pattern as `RPTreeForestIndex` and the LSH
multi-table build.

# Splitter choice is critical

Forests over PCA trees ONLY make sense with a *randomised* splitter
(at least one of the four extension axes must consume `rng` per call).
The default `splitter = pca_forest_splitter()` is the recommended
recipe — it randomises the spectrum estimator (`SubsampledSVD` +
`RandomizedSVD`) and the direction policy (`RandomTopK(3)`), so each
tree sees a different partition.

Building a forest with the deterministic `PCASplitter()` would silently
produce N identical trees and waste both build time and the parallel
union-of-buckets premise. The default guards against this; users who
override `splitter` should verify their splitter is randomised.
"""
struct PCATreeForestIndex{T<:AbstractFloat,D,Sp<:PCASplitter} <: AbstractANNIndex
    trees::Vector{PCATreeIndex{T,D,Sp}}
    dimension::Int
    n_points::Int
    distance::D
end

index_distance(index::PCATreeForestIndex) = index.distance
configured_k(::PCATreeForestIndex) = nothing
supports_mutation(::PCATreeForestIndex) = false

const PCATREE_FOREST_DEFAULT_N_TREES = 8

"""
    build_index(PCATreeForestIndex, data; n_trees, splitter, distance, rng)

Build a `PCATreeForestIndex` of `n_trees` independent PCA trees.

Each tree is built on its own `Threads.@threads`-scheduled task with
an RNG derived from `rng` via `spawn_child_rngs`. Per-tree seeds are
computed serially up front, so the resulting forest is deterministic
regardless of thread count or scheduling order.

The default `splitter = pca_forest_splitter()` is randomised across
spectrum and direction axes — see [`pca_forest_splitter`](@ref). Using
the deterministic `PCASplitter()` here would build N identical trees;
the docstring on [`PCATreeForestIndex`](@ref) explains why.
"""
function build_index(
    ::Type{PCATreeForestIndex},
    data::AbstractMatrix{T};
    n_trees::Integer = PCATREE_FOREST_DEFAULT_N_TREES,
    splitter::PCASplitter = pca_forest_splitter(),
    distance::D = default_distance,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:AbstractFloat,D}
    d, n = size(data)
    d > 0 || throw(ArgumentError("Dataset must have positive dimension"))
    n > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    n_trees > 0 || throw(ArgumentError("n_trees must be positive"))

    tree_rngs = spawn_child_rngs(rng, Int(n_trees))
    Sp = typeof(splitter)
    trees = Vector{PCATreeIndex{T,D,Sp}}(undef, Int(n_trees))
    Threads.@threads for t in 1:Int(n_trees)
        trees[t] = build_index(
            PCATreeIndex, data;
            splitter = splitter, distance = distance, rng = tree_rngs[t],
        )
    end
    return PCATreeForestIndex{T,D,Sp}(trees, d, n, distance)
end

"""
    query(index::PCATreeForestIndex, data, q, k)

Approximate kNN by routing `q` through every tree, taking the union of
leaf buckets as the candidate set, and brute-force scoring with
`index.distance`. Returns up to `k` neighbours sorted by distance.
Concurrent-safe: reads only from the index and uses a per-call scratch
BitSet plus heap.
"""
function query(
    index::PCATreeForestIndex{T,D,Sp},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer,
) where {T<:AbstractFloat,D,Sp}
    validate_index_dimensions(index, data, q)
    S = float(T)
    k <= 0 && return Neighbor{S}[]
    actual_k = min(Int(k), index.n_points)
    actual_k == 0 && return Neighbor{S}[]

    seen = BitSet()
    heap = BoundedMaxHeap{S}(actual_k)
    @inbounds for tree in index.trees
        members = _pcatree_route(tree, q)
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
