"""
MultiLevelIndex: Top-level wrapper for multi-level hierarchical indices.

This provides the public API for FAISS-like hierarchical ANN indices with
configurable transforms, routing, and merging strategies.
"""

"""
    MultiLevelIndex{T<:AbstractTransform, I<:AbstractANNIndex, D} <: AbstractANNIndex

Top-level multi-level index with configurable merge strategy and distance function.

This wraps a TransformedIndex tree and adds a global merge strategy for
combining results from multiple child indices.

# Fields
- `root::TransformedIndex{T,I}`: Root of the index tree
- `merge_strategy::AbstractMergeStrategy`: Strategy for merging results from multiple probes
- `distance::D`: Distance function used for recomputing distances from terminal indices

# Type Parameters
- `T`: Type of root transform
- `I`: Type of indices in root layer
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
struct MultiLevelIndex{T<:AbstractTransform, I<:AbstractANNIndex, D} <: AbstractANNIndex
    root::TransformedIndex{T,I}
    merge_strategy::AbstractMergeStrategy
    distance::D

    function MultiLevelIndex(
        root::TransformedIndex{T,I},
        merge_strategy::AbstractMergeStrategy,
        distance::D
    ) where {T<:AbstractTransform, I<:AbstractANNIndex, D}
        new{T, I, D}(root, merge_strategy, distance)
    end
end
