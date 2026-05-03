# ADR 0009: Multi-Level Index Architecture

## Status

Accepted

## Context

We need to implement hierarchical, FAISS-like multi-level indexing for approximate nearest neighbor (ANN) search. The system should support:

- Coarse-to-fine search with multiple levels of bucketing/quantization
- Composable transforms (KMeans, PQ, OPQ, PCA, etc.)
- Flexible routing strategies (top-K probing, exhaustive, etc.)
- High performance without sacrificing modularity

The design must cleanly separate:
1. Data transformation (e.g., quantization, clustering)
2. Routing decisions (which buckets to probe)
3. Index structures (HNSW, LSH, etc.)

## Decision

### 1. Recursive TransformedIndex Structure

We will use a **recursive tree structure** where each node is a `TransformedIndex` that can contain other `TransformedIndex` nodes or terminal indices.

```julia
struct TransformedIndex{
    T<:AbstractTransform,
    I<:AbstractANNIndex,
    C<:Union{Nothing, Vector{<:AbstractMatrix}},
    M<:Union{Nothing, Vector{Vector{Int}}},
} <: AbstractANNIndex
    transform::T
    routing_strategy::AbstractRoutingStrategy
    indices::Vector{I}      # Children (may be TransformedIndex or terminal)
    id_mappings::M          # Vector{Vector{Int}} for bucketing transforms; Nothing otherwise
    child_data::C           # Optional stored datasets when the transform changes representation
    bucket_lookup::Union{Nothing, Vector{Int}}  # original bucket id → position in `indices`
end

struct MultiLevelIndex{T,I,C,M,D} <: AbstractANNIndex
    root::TransformedIndex{T,I,C,M}
    merge_strategy::AbstractMergeStrategy
    distance::D
end
```

`C` and `M` are tracked as type parameters so the dispatch in `_resolve_child_data` and `_child_distance_type` can specialise per (storage, mapping) shape without runtime branches.

**Rationale**: Recursive structure provides maximum flexibility and composability. Each level encapsulates its own transform and routing logic.

**Alternative rejected**: Flat `Vector{TransformedIndex}` - less flexible, harder to represent arbitrary tree structures.

### 2. Transform Interface with Dual Output

All transforms implement:

```julia
abstract type AbstractTransform end

struct TransformResult{T,B}
    data::T          # Transformed representation (e.g., PQ codes, original vector)
    assignment::B    # Routing information (e.g., cluster distances, or nothing)
end

fit!(t::AbstractTransform, X::Matrix)
transform(t::AbstractTransform, x::AbstractVector)::TransformResult
```

Every transform **always returns both outputs**, even if one is trivial:
- `IdentityTransform`: `(x, nothing)`
- `KMeansTransform`: `(x, KMeansAssignment(distances))`
- `PQTransform`: `(pq_codes, nothing)`

Transforms also expose a `preserves_data(::AbstractTransform)` predicate. Transforms that only
add routing metadata (e.g., `IdentityTransform`, `KMeansTransform`) return `true`, signaling that
multi-level indices can reuse caller-provided data instead of caching redundant copies. Any
transform that produces a new representation keeps the default `false`, and the resulting
`TransformedIndex` stores the transformed datasets once during build so queries remain efficient.

**Rationale**: Uniform interface simplifies composition. No special-casing for "bucketing" vs "encoding" transforms.

**Alternative rejected**: Separate `AbstractBucketing` and `AbstractEncoding` types - creates artificial distinction and complicates composition.

### 3. KMeans Returns Full Distance Vector

`KMeansTransform` returns distances to **all centroids**, not just the nearest:

```julia
struct KMeansAssignment
    distances::Vector{Float32}  # Distance to each centroid
end
```

**Rationale**: Enables top-K routing without recomputing distances. Essential for multi-probe strategies.

**Alternative rejected**: Return single cluster ID - forces recomputation for top-K probing.

### 4. Pluggable Per-Level Routing Strategies

Routing is decoupled from transforms via strategy pattern:

```julia
abstract type AbstractRoutingStrategy end

struct TopKRouting <: AbstractRoutingStrategy
    k::Int
end

struct ExhaustiveRouting <: AbstractRoutingStrategy end

select_indices(strategy, assignment, indices)::Vector{AbstractANNIndex}
```

Each `TransformedIndex` has its own routing strategy.

**Rationale**: Maximum flexibility. Different levels can use different strategies (e.g., top-5 at L1, exhaustive at L2).

**Alternative rejected**: Hardcoded top-K or global strategy - inflexible, prevents experimentation.

### 5. Uniform Vector Storage for Children

`indices::Vector{I}` is **always a vector**, even when `length == 1`.

**Rationale**: Uniform interface simplifies code. No special-casing for single vs. multiple children.

**Alternative rejected**: `indices::Union{I, Vector{I}}` - requires constant type checking.

### 6. Parametric Types for Performance

`TransformResult{T,B}` and `TransformedIndex{T,I}` are parametric types.

**Rationale**: Type stability. Julia can specialize and eliminate dynamic dispatch.

**Alternative rejected**: Untyped fields or abstract types - performance penalty from dynamic dispatch.

### 7. Transforms Live in Separate Module

Transforms are organized in `src/transforms/` directory:

```
src/
  transforms/
    AbstractTransform.jl
    IdentityTransform.jl
    KMeansTransform.jl
    PQTransform.jl
  indices/
    multilevel/
      routing.jl
      transformed.jl
      multilevel.jl
```

**Rationale**: Transforms are conceptually independent from indices. Separation improves modularity and testability.

**Alternative rejected**: Inline transforms in `multilevel/` - creates tight coupling.

### 8. No Backtracking Across Levels

At each level, the routing strategy selects indices to probe. Once selected, search proceeds into those subtrees **without backtracking to sibling subtrees**.

**Rationale**: Matches FAISS behavior. Backtracking is expensive and rarely improves recall enough to justify cost.

**Note**: Individual indices (e.g., HNSW) may internally backtrack during their own search. This decision only applies to routing between levels.

**Alternative rejected**: Cross-level backtracking - added complexity with marginal benefit.

### 9. Pluggable Merge Strategy

When multiple indices are probed (e.g., top-K clusters), results are merged using a pluggable strategy:

```julia
abstract type AbstractMergeStrategy end

function merge_results(
    strategy::AbstractMergeStrategy,
    result_lists::Vector{Vector{Neighbor}},
    k::Int
)::Vector{Neighbor}
end
```

The merge strategy is stored as a field in `MultiLevelIndex` and applies globally to all merges.

**Rationale**: Different use cases require different merge semantics (trust sub-index distances vs. recompute with original data). Strategy pattern provides flexibility.

**Alternative rejected**: Hardcoded merge logic - inflexible, prevents experimentation.

### 10. Config-Based build_index with Automatic Fitting

Users provide a declarative config tree; `build_index` handles all transform fitting, data partitioning, and child index construction:

```julia
abstract type AbstractIndexConfig end

# Terminal index (leaf node)
struct TerminalConfig{I<:AbstractANNIndex, P} <: AbstractIndexConfig
    index_type::Type{I}
    params::P  # NamedTuple of parameters
end

# Transformed index (internal node)
struct TransformedConfig{T<:AbstractTransform, C<:AbstractIndexConfig} <: AbstractIndexConfig
    transform::T                      # Unfitted transform
    routing::AbstractRoutingStrategy
    child_config::C                   # Type-stable recursive config
end

# Main build_index entry point
function build_index(
    ::Type{MultiLevelIndex},
    X::Matrix,
    config::TransformedConfig;
    merge_strategy::AbstractMergeStrategy = SimpleMerge()
)
    root = _build_transformed(X, config)
    return MultiLevelIndex(root, merge_strategy)
end
```

**Example usage**:
```julia
# IVF: KMeans(100) → HNSW per cluster (with kmeans++ initialization)
config = TransformedConfig(
    KMeansTransform(k=100, distance=euclidean, init=:kmeans_plus_plus),
    TopKRouting(5),
    TerminalConfig(HNSWIndex, (M=16, ef_construction=200))
)

index = build_index(MultiLevelIndex, X, config)

# Alternative: naive random initialization
config_naive = TransformedConfig(
    KMeansTransform(k=100, distance=euclidean, init=:random),
    TopKRouting(5),
    TerminalConfig(HNSWIndex, (M=16, ef_construction=200))
)
```

**Rationale**:
- User-friendly: Single call, no manual fitting/partitioning
- Type-stable: Config tree is fully parametric
- Declarative: Structure is clear and explicit
- Consistent: Matches `build_index` pattern throughout ManifoldANN
- Flexible: Supports arbitrary depth via recursive `TransformedConfig`

**Alternative rejected**: Manual tree construction - too verbose, requires users to understand internal transform fitting and data partitioning logic.

### 11. Transforms Store Their Own Distance Functions

Each transform declares its distance function at construction time:

```julia
KMeansTransform(k=100, distance=euclidean, init=:kmeans_plus_plus)
PQTransform(M=8, nbits=8, distance=euclidean)
```

**Rationale**:
- Transforms may change the appropriate distance function for child indices (e.g., PQ changes to asymmetric distance)
- Each transform knows what metric it needs for its internal operations
- Explicit configuration prevents ambiguity

**Alternative rejected**: Infer distance from context - fragile, limits transform composability.

**Note on KMeans**: Custom implementation with both `:random` and `:kmeans_plus_plus` initialization strategies, based on JuliANN's kmeans implementation.

## Consequences

### Positive

- **Composable**: Arbitrary combinations of transforms and indices
- **Extensible**: New transforms/strategies added without modifying core
- **Performant**: Type stability, zero-cost abstractions
- **Clear separation of concerns**: Transform ≠ Routing ≠ Index
- **FAISS-like expressiveness**: Can replicate IVF, IMI, IVFPQ, etc.

### Negative

- **Initial complexity**: More abstractions than flat design (configs, transforms, routing, merge)
- **Learning curve**: Users must understand config tree structure
- **Memory overhead**: Each node stores routing strategy; MultiLevelIndex stores merge strategy

### Risks

- **Over-abstraction**: May discover constraints that don't fit cleanly
- **Mitigation**: Start with simple transforms (Identity, KMeans), iterate

### Implementation Notes

**Config Cloning (CRITICAL):**
When building child indices from a shared config, we must `deepcopy` the config for each child. TransformedConfig is immutable but holds references to mutable transform objects. Without deepcopy, all sibling nodes would share the same transform instance, and each `fit!` call would overwrite the previous partition's parameters, completely breaking multi-level hierarchies. See `src/indices/multilevel/builder.jl:92-101`.

### Open Questions

1. **Batch transforms**: Add `transform(t, X::Matrix)` for efficiency, or rely on single-vector `transform(t, x)` with iteration? - Deferred to optimization phase
2. **Concrete merge strategies**: What merge strategies should we implement initially? (SimpleMerge that trusts distances, RecomputeMerge that uses original data, etc.)
3. **Data partitioning utilities**: Where should `partition_by_transform(X, transform)` live? In transforms module or multilevel? - **DECIDED**: `src/transforms/utils.jl`
4. **KMeans implementation**: **DECIDED** - Implement custom KMeans with both `:random` (naive) and `:kmeans_plus_plus` initialization strategies. Reference implementation from JuliANN (`~/projects/mai/thesis/code/JuliANN/src/clustering/kmeans.jl`) includes Lloyd's algorithm, vectorized distance computation, and empty cluster handling.

## Implementation Plan

**Phase 1 (Initial):**
1. Config types (AbstractIndexConfig, TerminalConfig, TransformedConfig)
2. Transform interface + IdentityTransform
3. Routing strategies (TopK, Exhaustive) + select_indices
4. Merge strategies (SimpleMerge) + merge_results
5. TransformedIndex type + query logic
6. MultiLevelIndex wrapper
7. build_index implementation (_build_transformed, _build_from_config)
8. KMeansTransform (custom implementation)
   - Lloyd's algorithm with vectorized distance computation
   - `:random` (naive) initialization
   - `:kmeans_plus_plus` initialization
   - Empty cluster handling
   - Reference: JuliANN's kmeans implementation
9. Data partitioning utilities
10. Tests

**Phase 2 (Future):**
- PQTransform
- OPQTransform
- Additional merge strategies
- Batch transform optimizations
- Helper builders for common patterns

## References

- [FAISS Wiki: Inverted File Index](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes#inverted-file-index)
- Implementation plan: See `multi_index_plan.md` in project root
