import Base: insert!

"""
Core interface for approximate nearest-neighbor (ANN) indices.

All concrete indices must implement `build_index` and `query` while keeping
their node/edge payloads type-stable so Julia can specialize the critical
paths. Data matrices are always supplied at query time to avoid duplicating
storage and to allow callers to provide alternative representations
(denoised, projected, etc.).
"""
abstract type AbstractANNIndex end

"""
ANN index that can expose an explicit `KNNGraph`.

Implementations must provide `materialize_graph` along with capability
introspection methods describing what graph features they support
(custom layers, node metadata, mutation, …).
"""
abstract type AbstractGraphIndex <: AbstractANNIndex end

"""
    build_index(::Type{T}, data; kwargs...) -> T

Construct an ANN index of type `T` from `data`. Builders should avoid holding
onto `data` directly; they store only the structural metadata needed for
search. Keyword arguments configure algorithm-specific parameters.
"""
function build_index end

"""
    query(index, data, q, k; kwargs...) -> Vector{Neighbor}

Return up to `k` approximate nearest neighbors of query point `q`. Each entry
stores both the identifier and the distance the index computed internally.
The backing dataset `data` is always supplied explicitly, which allows the
same index to serve multiple data representations (e.g. PCA-denoised copies).
Concrete indices handle algorithm-specific keyword arguments such as search
policies.
"""
function query end

"""
    materialize_graph(index::AbstractGraphIndex)

Produce a `KNNGraph` representing the index connectivity. Used by downstream
graph algorithms (shortest paths, diffusion, etc.). Indices that do not natively
store graphs may construct one on demand.
"""
function materialize_graph end

"""
    ensure_graph(index::AbstractANNIndex)

Return a `KNNGraph` for `index` if possible, otherwise throw an informative
error. Useful when downstream code requires an explicit graph regardless of
how the index stores its structure internally.
"""
function ensure_graph(index::AbstractANNIndex)
    if index isa AbstractGraphIndex
        return materialize_graph(index)
    end
    throw(ArgumentError("Index $(typeof(index)) does not expose a kNN graph"))
end

"""
    configured_k(index::AbstractANNIndex) -> Union{Int, Nothing}

Return the neighbor count that `index` was configured with, if any. Graph
construction utilities use this to validate `k` requests so callers do not
ask for denser graphs than the index can support. Defaults to `nothing`
meaning "no preference/limit".
"""
configured_k(::AbstractANNIndex) = nothing

# Distance helper -------------------------------------------------------------

"""
    index_distance(index::AbstractANNIndex) -> Union{Nothing, Function}

Return the distance function intrinsically associated with `index`, if any.
Indices that store their metric (e.g. HNSW, BruteForce) should override this
to expose it. The default `nothing` signals that callers should fall back to
their own distance function when recomputing neighbor scores.
"""
index_distance(::AbstractANNIndex) = nothing

# Capability introspection hooks let algorithms branch cheaply without repeated
# type checks or ad-hoc keyword plumbing.
"""
    supports_layers(index::AbstractANNIndex) -> Bool

Whether `index` exposes explicit hierarchy or layer control (e.g. HNSW top
layers). Defaults to `false`.
"""
supports_layers(::AbstractANNIndex) = false

"""
    supports_metadata(index::AbstractANNIndex) -> Bool

Whether `index` guarantees node metadata beyond point identifiers (such as
tangent information or PCA projections). Defaults to `false`.
"""
supports_metadata(::AbstractANNIndex) = false

"""
    supports_mutation(index::AbstractANNIndex) -> Bool

Whether `index` supports inserting/removing points after construction.
Mutable indices must specify how metadata stays consistent. Defaults to `false`.
"""
supports_mutation(::AbstractANNIndex) = false

"""
    insert!(index::AbstractANNIndex, args...; kwargs...)

Optional mutation hook (extending `Base.insert!`) for indices that allow
dynamic updates. Implementations must document the accepted arguments
(typically the new point or batch plus any metadata). The default method
throws to signal immutability.
"""
function insert!(index::AbstractANNIndex, args...; kwargs...)
    throw(ArgumentError("Index $(typeof(index)) does not support mutation"))
end
