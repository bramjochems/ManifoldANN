"""
MultiLevelIndex: Top-level wrapper for multi-level hierarchical indices.

This provides the public API for FAISS-like hierarchical ANN indices with
configurable transforms, routing, and merging strategies.
"""

"""
    MultiLevelIndex{T<:AbstractTransform, I<:AbstractANNIndex, C, M, D} <: AbstractANNIndex

Top-level multi-level index with configurable merge strategy and distance function.

This wraps a TransformedIndex tree and adds a global merge strategy for
combining results from multiple child indices.

# Fields
- `root::TransformedIndex{T,I,C,M}`: Root of the index tree
- `merge_strategy::AbstractMergeStrategy`: Strategy for merging results from multiple probes
- `distance::D`: Distance function used for recomputing distances from terminal indices

# Type Parameters
- `T`: Type of root transform
- `I`: Type of indices in root layer
- `C`: Type of stored child datasets at the root (either `Nothing` or `Vector{<:AbstractMatrix}`)
- `M`: Type of id mappings at the root (either `Nothing` or `Vector{Vector{Int}}`)
- `D`: Type of distance function

# Examples
```julia
# Build IVF index
config = TransformedConfig(
    KMeansTransform(k=100, distance=Euclidean()),
    TopKRouting(5),
    TerminalConfig(HNSWIndex, (M=16, ef_construction=200))
)

index = build_index(MultiLevelIndex, X, config)

# Query
q = rand(Float32, size(X, 1))
neighbors = query(index, X, q, 10)
```
"""
struct MultiLevelIndex{T<:AbstractTransform, I<:AbstractANNIndex, C, M, D} <: AbstractANNIndex
    root::TransformedIndex{T,I,C,M}
    merge_strategy::AbstractMergeStrategy
    distance::D

    function MultiLevelIndex(
        root::TransformedIndex{T,I,C,M},
        merge_strategy::AbstractMergeStrategy,
        distance::D
    ) where {T<:AbstractTransform, I<:AbstractANNIndex, C, M, D}
        new{T, I, C, M, D}(root, merge_strategy, distance)
    end
end
