# Transforms for Multi-Level Indices

The transform system provides the foundation for hierarchical ANN indices in ManifoldANN. Transforms enable FAISS-like index structures such as IVF (Inverted File), multi-level clustering, and future quantization-based methods.

## Overview

Transforms operate at internal nodes of a multi-level index tree. Each transform:
1. **Learns parameters** during `fit!` (e.g., cluster centroids, codebooks)
2. **Processes query points** during search via `transform()`
3. **Returns dual output**: transformed data + routing information

The routing information guides which child indices to probe, enabling efficient coarse-to-fine search.

## Core Interface

All transforms implement the `AbstractTransform` interface:

```julia
abstract type AbstractTransform end

struct TransformResult{T,B}
    data::T          # Transformed representation
    assignment::B    # Routing information (or nothing)
end

# Required methods
fit!(transform::AbstractTransform, X::Matrix)
transform(transform::AbstractTransform, x::Vector)::TransformResult

# Optional trait
preserves_data(transform::AbstractTransform)::Bool
```

The `preserves_data` predicate defaults to `false`. Transforms that simply attach routing
metadata (e.g., `IdentityTransform`, `KMeansTransform`) override it to return `true`, which
allows multi-level indices to reuse caller-provided datasets instead of caching redundant
partitions. Any transform that produces a new representation should leave the default so
the transformed data is materialized once during build and reused for all queries.

## Available Transforms

### IdentityTransform

Pass-through transform that returns data unchanged with no bucketing.

**Use cases:**
- Testing multi-level infrastructure
- Creating a single-child wrapper around a terminal index
- Mixed hierarchies where some levels don't transform data

**Example:**
```julia
t = IdentityTransform()
fit!(t, X)  # No-op

x = rand(Float32, 128)
result = transform(t, x)
# result.data === x
# result.assignment === nothing
```

**Configuration:**
```julia
config = TransformedConfig(
    IdentityTransform(),
    ExhaustiveRouting(),  # Only one child, so probe it always
    TerminalConfig(HNSWIndex, (M=16,))
)
```

### KMeansTransform

Cluster-based partitioning for IVF (Inverted File) style indices.

**Parameters:**
- `k::Int`: Number of clusters
- `distance::SemiMetric`: Distance metric (from Distances.jl)
- `init::Symbol`: Initialization strategy
  - `:kmeans_plus_plus` (default): D² sampling for better initialization
  - `:random`: Uniform random centroid selection
- `max_iters::Int`: Maximum Lloyd iterations (default: 100)
- `tol::Float64`: Convergence tolerance (default: 1e-6)

**Algorithm:**
- Uses Lloyd's algorithm with vectorized distance computation
- Handles empty clusters by reassigning to farthest point
- Returns distances to **all** centroids (enables top-K routing)

**Example:**
```julia
using Distances

t = KMeansTransform(
    k=100,
    distance=Euclidean(),
    init=:kmeans_plus_plus
)

# Fit learns 100 cluster centroids
fit!(t, X)

# Transform returns original data + distances to all centroids
x = rand(Float32, 128)
result = transform(t, x)
# result.data === x (unchanged)
# result.assignment::KMeansAssignment with distances vector
```

**IVF Configuration:**
```julia
# Classic IVF: KMeans → HNSW per cluster
config = TransformedConfig(
    KMeansTransform(k=100, distance=Euclidean()),
    TopKRouting(5),  # Probe 5 nearest clusters
    TerminalConfig(HNSWIndex, (M=16, ef_construction=200))
)

index = build_index(MultiLevelIndex, X, config)
```

**Initialization Strategies:**

*KMeans++ (recommended):*
- First centroid: uniform random
- Subsequent: probability ∝ D(x)² to nearest centroid
- Better separation, faster convergence
- Reference: Arthur & Vassilvitskii (2007)

*Random:*
- Uniform random selection of k points
- Faster initialization, may need more Lloyd iterations
- Useful for experimentation or when initialization cost dominates

**Performance Notes:**
- Vectorized distance computation for efficiency
- Empty cluster handling adds negligible overhead
- For very large k (>1000), initialization can dominate build time

## Routing Strategies

Routing strategies determine which child indices to probe based on transform output.

### TopKRouting

Probe k nearest clusters/partitions based on assignment distances.

```julia
TopKRouting(k::Int)
```

**Use with:** `KMeansTransform` or any transform providing distance-based assignment

**Trade-offs:**
- Lower k: faster queries, lower recall
- Higher k: higher recall, more computation
- Common values: 1 (single probe), 5-10 (multi-probe IVF), 20+ (high recall)

**Example:**
```julia
# Multi-probe IVF: probe 10 nearest clusters
TransformedConfig(
    KMeansTransform(k=200, ...),
    TopKRouting(10),  # 5% of clusters
    TerminalConfig(...)
)
```

### ExhaustiveRouting

Probe all child indices (no pruning).

```julia
ExhaustiveRouting()
```

**Use with:**
- `IdentityTransform` (typically only one child anyway)
- Debugging and baseline comparisons
- Small number of children where pruning doesn't help

**Example:**
```julia
# Identity pass-through with exhaustive search
TransformedConfig(
    IdentityTransform(),
    ExhaustiveRouting(),
    TerminalConfig(HNSWIndex, (...))
)
```

## Merge Strategies

Merge strategies combine results from multiple probed child indices.

### SimpleMerge

Trust sub-index distances, sort globally, deduplicate.

```julia
SimpleMerge()
```

**Behavior:**
1. Collect all neighbors from all probed children
2. Sort by distance (as reported by child indices)
3. Remove duplicates (keep first = closest)
4. Return top-k

**Pros:**
- Fast: no distance recomputation
- Simple: no additional data access

**Cons:**
- May be inaccurate if children use approximate distances (e.g., future PQ)
- Distance semantics may differ across children

**Future:**
- `RecomputeMerge`: Recompute distances with original data for accuracy
- `ResidualMerge`: Use residual distances for PQ-based indices

## Multi-Level Hierarchies

Transforms can be nested arbitrarily deep by using `TransformedConfig` as the child config.

### Two-Level KMeans

```julia
# Coarse clustering → fine clustering → HNSW
config = TransformedConfig(
    KMeansTransform(k=100, ...),     # 100 coarse clusters
    TopKRouting(10),
    TransformedConfig(
        KMeansTransform(k=20, ...),  # 20 fine clusters per coarse
        TopKRouting(3),
        TerminalConfig(HNSWIndex, (...))
    )
)
```

**Result:**
- Top level: 100 coarse clusters
- Second level: 100 × 20 = 2000 fine clusters total
- Query probes: 10 × 3 = 30 fine clusters

**Use cases:**
- Very large datasets (10M+ points)
- Hierarchical exploration (coarse → fine)
- Research on multi-level partitioning strategies

### Mixed Hierarchies

Combine different transform types at different levels:

```julia
# KMeans → Identity → HNSW
# (Useful for debugging or specific research questions)
config = TransformedConfig(
    KMeansTransform(k=50, ...),
    TopKRouting(5),
    TransformedConfig(
        IdentityTransform(),
        ExhaustiveRouting(),
        TerminalConfig(HNSWIndex, (...))
    )
)
```

## Configuration Patterns

### IVF-Flat (KMeans → Brute Force)

Fast training, exact distances within probed clusters:

```julia
TransformedConfig(
    KMeansTransform(k=100, distance=Euclidean()),
    TopKRouting(5),
    TerminalConfig(BruteForceIndex, ())
)
```

### IVF-HNSW (KMeans → HNSW)

Balanced speed/accuracy, graph-based refinement:

```julia
TransformedConfig(
    KMeansTransform(k=100, distance=Euclidean()),
    TopKRouting(5),
    TerminalConfig(HNSWIndex, (M=16, ef_construction=200))
)
```

### IVF-LSH (KMeans → LSH)

High-dimensional data with hierarchical hashing:

```julia
TransformedConfig(
    KMeansTransform(k=50, distance=Euclidean()),
    TopKRouting(10),
    TerminalConfig(LSHIndex, (n_tables=10, hash_length=16))
)
```

## Design Rationale

See `docs/design/ADR-0009-multilevel-index-architecture.md` for detailed architectural decisions:

- **Why dual output transforms?** Uniform interface, no special-casing
- **Why KMeans returns full distance vector?** Enables efficient top-K routing
- **Why parametric types?** Type stability for performance
- **Why separate transform module?** Modularity, reusability, testability
- **Why config-based builder?** User-friendly, declarative, type-safe

## Future Transforms

### PQTransform (Product Quantization)

Compress vectors into short codes for memory efficiency:

```julia
# Not yet implemented
PQTransform(
    M=8,           # Number of subquantizers
    nbits=8,       # Bits per subquantizer (256 centroids)
    distance=...
)
```

**Use case:** IVFPQ index (most popular FAISS index)

### OPQTransform (Optimized PQ)

Learn rotation before PQ for better quantization:

```julia
# Not yet implemented
OPQTransform(
    M=8,
    nbits=8,
    n_iter=20,     # Rotation optimization iterations
    distance=...
)
```

### PCATransform

Dimensionality reduction before downstream index:

```julia
# Not yet implemented
PCATransform(n_components=64)
```

## Performance Considerations

### KMeans Build Time

- Dominated by Lloyd iterations and data partitioning
- Initialization cost: O(nk) for random, O(nk²) for kmeans++
- Lloyd iterations: O(nkd) per iteration
- For k=100, d=128, n=1M: ~2-10 seconds on modern CPU

### Query Time

- Transform overhead: O(kd) for KMeans distance computation
- Routing overhead: O(k) for top-K selection
- Typically negligible vs child index query time
- Example: 1-2ms transform + 5-50ms HNSW = ~1-4% overhead

### Memory Usage

- KMeans: O(kd) for centroids (typically KB-MB)
- Config tree: Negligible
- Dominant cost: Child indices

## See Also

- `docs/indices.md` - Multi-level index usage
- `docs/examples/indices/06-ivf-index.jl` - Basic IVF example
- `docs/examples/indices/07-multilevel-hierarchy.jl` - Two-level clustering
- `docs/design/ADR-0009-multilevel-index-architecture.md` - Architecture decisions
- `multi_index_plan.md` - Implementation plan (root directory)
