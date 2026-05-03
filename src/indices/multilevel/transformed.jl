"""
TransformedIndex: Internal node in a multi-level index tree.

Each TransformedIndex applies a transform to incoming queries, routes to
child indices based on the transform's assignment, and collects results.
"""

"""
    TransformedIndex{T<:AbstractTransform, I<:AbstractANNIndex, C, M} <: AbstractANNIndex

An index that applies a transform before routing to child indices.

This is the recursive building block for multi-level indices. Each TransformedIndex can
contain other TransformedIndex nodes or terminal indices, forming an arbitrary tree structure.

# Fields
- `transform::T`: Fitted transform (e.g., KMeans with learned centroids)
- `routing_strategy::AbstractRoutingStrategy`: Strategy for selecting children to probe
- `indices::Vector{I}`: Child indices (may be TransformedIndex or terminal indices)
- `id_mappings::M`: Maps local IDs to global IDs for bucketing transforms (either `Nothing` or `Vector{Vector{Int}}`)
- `child_data::C`: Optional per-child datasets (only stored when the transform changes the representation; either `Nothing` or `Vector{<:AbstractMatrix}`)
- `bucket_lookup::Union{Nothing, Vector{Int}}`: Maps original bucket ids to positions in `indices`

# Type Parameters
- `T`: Type of transform
- `I`: Type of child indices (may be Union type for mixed children)
- `C`: Type of stored child datasets (either `Nothing` or `Vector{<:AbstractMatrix}`)
- `M`: Type of id mappings (either `Nothing` or `Vector{Vector{Int}}`)

# Examples
```julia
# IVF structure: KMeans transform with 100 HNSW children
kmeans = KMeansTransform(k=100, distance=Euclidean())
fit!(kmeans, X)
hnsw_indices = [build_index(HNSWIndex, partition; M=16) for partition in partitions]

ivf = TransformedIndex(kmeans, TopKRouting(5), hnsw_indices, id_mappings)
```
"""
struct TransformedIndex{
    T<:AbstractTransform,
    I<:AbstractANNIndex,
    C<:Union{Nothing, Vector{<:AbstractMatrix}},
    M<:Union{Nothing, Vector{Vector{Int}}},
} <: AbstractANNIndex
    transform::T
    routing_strategy::AbstractRoutingStrategy
    indices::Vector{I}
    id_mappings::M
    child_data::C
    bucket_lookup::Union{Nothing, Vector{Int}}

    function TransformedIndex(
        transform::T,
        routing_strategy::AbstractRoutingStrategy,
        indices::Vector{I},
        id_mappings::M=nothing,
        child_data::C=nothing,
        bucket_lookup::Union{Nothing, Vector{Int}}=nothing,
    ) where {
        T<:AbstractTransform,
        I<:AbstractANNIndex,
        C<:Union{Nothing, Vector{<:AbstractMatrix}},
        M<:Union{Nothing, Vector{Vector{Int}}},
    }
        isempty(indices) && throw(ArgumentError("Must have at least one child index"))
        new{T, I, C, M}(
            transform,
            routing_strategy,
            indices,
            id_mappings,
            child_data,
            bucket_lookup,
        )
    end
end
