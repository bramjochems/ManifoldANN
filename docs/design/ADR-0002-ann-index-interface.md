# ADR-0002: ANN Index Interface and Graph Trait


## Context

The package must support diverse ANN builders (brute force, KD trees, HNSW variants, LSH) and explicit kNN graphs that store manifold-specific metadata. We also need hooks for preprocessing (e.g. PCA, tangent estimation) and want the Julia compiler to specialize effectively by keeping element and metadata types concrete. The interface should avoid duplicating data, support arbitrary query points, and make it simple to materialize graphs when downstream algorithms need them.

## Decision

1. **Unified API**: Every index implements `build_index(::Type{<:AbstractANNIndex}, data; kwargs...)` and `query(index, data, q, k; kwargs...)`. Queries always pass the data matrix explicitly so indices only store structural metadata, letting callers swap in denoised or projected versions without rebuilding.
2. **Graph Trait**: Introduce `AbstractGraphIndex <: AbstractANNIndex` for indices that can expose an explicit `KNNGraph`. Provide `materialize_graph(index)` (or `ensure_graph(index)`) plus capability predicates (`supports_layers`, `supports_metadata`) so algorithms can branch cheaply.
3. **Metadata Discipline**: Node payloads use parametric structs (`GraphNode{T,Dim,Meta}`) with typed `Meta` slots to hold tangents, PCA coefficients, or `NoMetadata`. Builders declare their metadata requirements via traits, keeping dispatch predictable.
4. **Search Policies**: Tuning knobs (beam width, backtracking limits, multi-probe schedules) are modeled as separate policy objects per index family, keeping constructors lean and making traversal strategies interchangeable.
5. **Projection Flexibility**: Projection-based indices (HNSW, LSH, RP trees) may use random hyperplanes, tangent-aligned directions, or random combinations of eigenvectors without forcing a tree structure. LSH remains table-based with multi-probe support; trees are only used when hierarchical splits are deliberately needed.

## Consequences

- Data externalization avoids memory duplication and allows experimenting with different point representations at query time.
- Any ANN structure can serve graph-centric workflows by implementing the graph trait, but pure graph builders remain free to focus on explicit adjacency construction.
- Concrete metadata types keep Julia’s specialization strong and make BLAS-friendly kernels easier to optimize.
- Variants of HNSW or other indices share core logic yet offer clearly named constructors, since behaviors are driven by policies and typed metadata rather than large keyword lists.
- Multi-probe LSH and eigenvector-based projections fit naturally into the existing API without requiring entirely new storage abstractions.
