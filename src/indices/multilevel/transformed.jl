"""
TransformedIndex: Internal node in a multi-level index tree.

Each TransformedIndex applies a transform to incoming queries, routes to
child indices based on the transform's assignment, and collects results.
"""

"""
    TransformedIndex{T<:AbstractTransform, I<:AbstractANNIndex} <: AbstractANNIndex

An index that applies a transform before routing to child indices.

This is the recursive building block for multi-level indices. Each TransformedIndex can
contain other TransformedIndex nodes or terminal indices, forming an arbitrary tree structure.

# Fields
- `transform::T`: Fitted transform (e.g., KMeans with learned centroids)
- `routing_strategy::AbstractRoutingStrategy`: Strategy for selecting children to probe
- `indices::Vector{I}`: Child indices (may be TransformedIndex or terminal indices)
- `id_mappings::Union{Nothing, Vector{Vector{Int}}}`: Maps local IDs to global IDs for bucketing transforms
- `partition_data::Union{Nothing, Vector{Matrix}}`: Partition data for bucketing transforms (for querying children)

# Type Parameters
- `T`: Type of transform
- `I`: Type of child indices (may be Union type for mixed children)

# Examples
```julia
# IVF structure: KMeans transform with 100 HNSW children
kmeans = KMeansTransform(k=100, distance=Euclidean())
fit!(kmeans, X)
hnsw_indices = [build_index(HNSWIndex, partition; M=16) for partition in partitions]

ivf = TransformedIndex(kmeans, TopKRouting(5), hnsw_indices, id_mappings)
```
"""
struct TransformedIndex{T<:AbstractTransform, I<:AbstractANNIndex} <: AbstractANNIndex
    transform::T
    routing_strategy::AbstractRoutingStrategy
    indices::Vector{I}
    id_mappings::Union{Nothing, Vector{Vector{Int}}}
    partition_data::Union{Nothing, Vector}  # Vector of matrices (any element type)

    function TransformedIndex(
        transform::T,
        routing_strategy::AbstractRoutingStrategy,
        indices::Vector{I},
        id_mappings::Union{Nothing, Vector{Vector{Int}}}=nothing,
        partition_data::Union{Nothing, Vector}=nothing
    ) where {T<:AbstractTransform, I<:AbstractANNIndex}
        isempty(indices) && throw(ArgumentError("Must have at least one child index"))
        new{T, I}(transform, routing_strategy, indices, id_mappings, partition_data)
    end
end
