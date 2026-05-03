"""
Configuration types for multi-level indices.

These types provide a declarative way to specify the structure of hierarchical
indices. The config tree is fully parametric for type stability.
"""

"""
    AbstractIndexConfig

Abstract base type for index configurations.

Concrete subtypes:
- `TerminalConfig{I,P}`: Leaf node configuration (builds a concrete index)
- `TransformedConfig{T,C}`: Internal node configuration (applies transform + routing)
"""
abstract type AbstractIndexConfig end

"""
    TerminalConfig{I<:AbstractANNIndex, P}

Configuration for a terminal (leaf) index in a multi-level structure.

# Fields
- `index_type::Type{I}`: Type of index to build (e.g., `HNSWIndex`)
- `params::P`: NamedTuple of parameters to pass to `build_index`

# Examples
```julia
# HNSW terminal with specific parameters
config = TerminalConfig(HNSWIndex, (M=16, ef_construction=200))

# Graph-based terminal
config = TerminalConfig(NNDescentIndex, (k=20, max_iters=10))
```
"""
struct TerminalConfig{I<:AbstractANNIndex, P} <: AbstractIndexConfig
    index_type::Type{I}
    params::P  # NamedTuple

    function TerminalConfig(index_type::Type{I}, params::P) where {I<:AbstractANNIndex, P}
        new{I, P}(index_type, params)
    end
end

"""
    TransformedConfig{T<:AbstractTransform, C<:AbstractIndexConfig}

Configuration for a transformed index (internal node) in a multi-level structure.

# Fields
- `transform::T`: Prototype transform (declarative, never mutated by build).
  `build_index` deepcopies this once per `TransformedIndex` it constructs and
  fits the copy on that index's slice of data; the config object stays reusable.
- `routing::AbstractRoutingStrategy`: Strategy for selecting child indices to probe
- `child_config::C`: Configuration for child indices (may be TerminalConfig or another TransformedConfig)

# Examples
```julia
# IVF: KMeans → HNSW per cluster
config = TransformedConfig(
    KMeansTransform(k=100, distance=Euclidean(), init=:kmeans_plus_plus),
    TopKRouting(5),
    TerminalConfig(HNSWIndex, (M=16, ef_construction=200))
)

# Two-level: KMeans → Identity → HNSW
config = TransformedConfig(
    KMeansTransform(k=100, distance=Euclidean()),
    TopKRouting(5),
    TransformedConfig(
        IdentityTransform(),
        ExhaustiveRouting(),
        TerminalConfig(HNSWIndex, (M=16,))
    )
)
```
"""
struct TransformedConfig{T<:AbstractTransform, C<:AbstractIndexConfig} <: AbstractIndexConfig
    transform::T
    routing::AbstractRoutingStrategy
    child_config::C

    function TransformedConfig(
        transform::T,
        routing::AbstractRoutingStrategy,
        child_config::C
    ) where {T<:AbstractTransform, C<:AbstractIndexConfig}
        new{T, C}(transform, routing, child_config)
    end
end
