# ADR-0012: Flexible Distance Metrics for ORC Computation

## Context

The original Ollivier-Ricci curvature implementation used Euclidean distance for
both the OT cost matrix and the curvature denominator:

```julia
κ(x,y) = 1 - W₁(μₓ, μᵧ) / ||x - y||₂
```

This works well for ambient space, but fails for manifold learning where:
1. **Geodesic distances** better capture manifold structure than Euclidean
2. **Scale normalization** (effective_epsilon) provides invariance to local density
3. Different distance choices in numerator vs denominator enable different ORC variants

The orcml paper (ICLR 2025) uses:
- Cost matrix: Geodesic distances with effective_epsilon edge weights
- Denominator: effective_epsilon (not Euclidean distance)

We need a flexible system to support multiple ORC computation strategies.

## Decision

1. **Two Distance Parameters**:
   ```julia
   compute_all_curvatures(
       graph, data;
       cost_metric = :euclidean,        # For OT cost matrix
       denominator_metric = :euclidean  # For curvature denominator
   )
   ```

2. **Supported Metrics**:
   - `:euclidean`: Direct Euclidean distance ||i - j||
   - `:geodesic_euclidean`: Shortest path with Euclidean edge weights
   - `:geodesic_normalized`: Shortest path with effective_epsilon edge weights (orcml)
   - `:geodesic_unit`: Shortest path with unit edge weights (hop count)
   - `:normalized`: effective_epsilon only (for denominator)

3. **Distance Function Abstraction**:
   ```julia
   get_distance_function(metric::Symbol, graph, data) -> (i, j) -> Float64
   ```
   - Pre-computes shortest paths for geodesic metrics (O(n³) Floyd-Warshall)
   - Returns closure for fast per-edge access (O(1) lookup)
   - Handles both directed and undirected graphs

4. **effective_epsilon Implementation**:
   - Computes mean k-NN distance for scale normalization
   - Uses `original_k` from metadata (important for undirected graphs)
   - Excludes edge endpoints to avoid bias (orcml convention)

5. **Validation**:
   ```julia
   validate_geodesic_config(graph, metric)
   ```
   - Warns when using geodesic metrics on directed graphs
   - Not an error - both work, but have different interpretations:
     - Directed: Asymmetric shortest paths (may have d(i,j) ≠ d(j,i))
     - Undirected: Symmetric shortest paths (guaranteed d(i,j) = d(j,i))
   - Helps users understand the implications of their choice

## Consequences

**Benefits**:
- **Replicate orcml**: Exact match with reference implementation
- **Research flexibility**: Compare different ORC formulations (Euclidean vs geodesic)
- **Intrinsic geometry**: Geodesic distances capture graph-based proximity
- **Scale invariance**: effective_epsilon handles varying local densities

**Performance**:
- Geodesic metrics: O(n³) pre-computation, O(1) per edge
- Euclidean metrics: O(d) per edge, no pre-computation
- Memory: geodesic requires O(n²) distance matrix

**Configuration Examples**:

```julia
# Original ManifoldANN (default)
compute_all_curvatures(graph, data)

# orcml replication
graph = build_knn_graph(index, data; k=15, directed=false)
compute_all_curvatures(graph, data;
    cost_metric=:geodesic_normalized,
    denominator_metric=:normalized,
    exclude_edge_endpoints=true
)

# Hybrid: geodesic costs, Euclidean denominator
compute_all_curvatures(graph, data;
    cost_metric=:geodesic_euclidean,
    denominator_metric=:euclidean
)
```

**Deprecation**:
- `distance_fn` parameter still supported for backward compatibility
- Issues warning to migrate to new API

## Future Work

- Add lazy geodesic computation (compute on-demand, not all-pairs)
- Explore alternative scale normalizations (median, max, etc.)
- Support custom distance functions via metric interface
- Profile memory usage for large graphs with geodesic metrics

## References

- orcml: https://github.com/TristanSaidi/orcml (effective_epsilon implementation)
- "Recovering Manifold Structure Using Ollivier-Ricci Curvature" (ICLR 2025)
- Ollivier, Y. (2009). "Ricci curvature of Markov chains on metric spaces"
