"""
Multi-level hierarchical indices for approximate nearest neighbor search.

This module implements FAISS-like hierarchical indexing with:
- Recursive TransformedIndex tree structure
- Pluggable transforms (KMeans, Identity, future: PQ, OPQ)
- Pluggable routing strategies (TopK, Exhaustive)
- Pluggable merge strategies (SimpleMerge, future: RecomputeMerge)

# Main Types
- `MultiLevelIndex`: Top-level index with merge strategy
- `TransformedIndex`: Internal node with transform + routing + children
- `AbstractIndexConfig`, `TerminalConfig`, `TransformedConfig`: Config types

# Main Functions
- `build_index(MultiLevelIndex, X, config; merge_strategy)`: Build index from config
- `query(index, data, q, k)`: Query for k nearest neighbors

# Exports
- Config types: `AbstractIndexConfig`, `TerminalConfig`, `TransformedConfig`
- Index types: `MultiLevelIndex`, `TransformedIndex`
- Routing: `AbstractRoutingStrategy`, `TopKRouting`, `ExhaustiveRouting`
- Merging: `AbstractMergeStrategy`, `SimpleMerge`

# Examples
```julia
using ManifoldANN
using Distances

# Build IVF index: KMeans(100) → HNSW per cluster
config = TransformedConfig(
    KMeansTransform(k=100, distance=Euclidean(), init=:kmeans_plus_plus),
    TopKRouting(5),
    TerminalConfig(HNSWIndex, (M=16, ef_construction=200))
)

X = rand(Float32, 128, 10000)
index = build_index(MultiLevelIndex, X, config)

# Query
q = rand(Float32, 128)
neighbors = query(index, X, q, 10)
```
"""

# Import transform utilities from parent module
using ...ManifoldANN: AbstractTransform, TransformResult, KMeansAssignment
using ...ManifoldANN: has_bucketing, partition_by_transform, apply_transform_batch
using ...ManifoldANN: fit!, transform

# Import parent module types
using ...ManifoldANN: AbstractANNIndex, build_index, query, Neighbor, index_distance

# Include module components (order matters for dependencies)
include("routing.jl")      # Defines AbstractRoutingStrategy
include("merge.jl")        # Defines AbstractMergeStrategy
include("config.jl")       # Uses AbstractRoutingStrategy
include("utils.jl")
include("transformed.jl")  # Uses AbstractTransform, AbstractRoutingStrategy
include("multilevel.jl")   # Uses TransformedIndex, AbstractMergeStrategy
include("builder.jl")      # Uses all config and index types
include("ivf_hnsw.jl")
include("query.jl")        # Uses all index types

# Exports
export AbstractIndexConfig, TerminalConfig, TransformedConfig
export AbstractRoutingStrategy, TopKRouting, ExhaustiveRouting
export AbstractMergeStrategy, SimpleMerge
export TransformedIndex, MultiLevelIndex
export build_ivf_hnsw_index
