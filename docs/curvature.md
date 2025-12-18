# Ollivier-Ricci Curvature (ORC)

This package provides a complete implementation of Ollivier-Ricci curvature computation for graph refinement.

## Overview

Ollivier-Ricci curvature measures how well edges follow manifold structure:
- **Positive curvature**: Edge follows the manifold (keep it)
- **Negative curvature**: Edge cuts across the manifold (remove it)

By filtering low-curvature edges, we can improve geodesic distance estimates on k-NN graphs.

## Quick Start

### Manifold Learning (Recommended)

For manifold learning with geodesic distances, use **undirected graphs**:

```julia
using ManifoldANN

# Build undirected graph (symmetrized for manifold structure)
index = build_index(NNDescentIndex, data; k=20)
graph = build_knn_graph(index, data; k=20, directed=false)

# Compute curvatures with geodesic distances
curvatures = compute_all_curvatures(
    graph, data;
    cost_metric=:geodesic_euclidean,
    solver=HungarianSolver(),
    fallback_solver=NetworkSimplexSolver()
)

# Filter low-curvature edges
filtered_graph = filter_graph(
    graph, data;
    curvatures=curvatures,
    curvature_threshold=0.0,
    min_neighbors=5
)
```

### Fast Local Curvature (No Geodesics)

For Euclidean distances without geodesics, directed graphs work fine:

```julia
# Build directed graph (default, k exactly preserved)
graph = build_knn_graph(index, data; k=20)  # directed=true by default

# Compute curvatures with direct Euclidean distances
curvatures = compute_all_curvatures(
    graph, data;
    cost_metric=:euclidean,  # No geodesics, directed is OK
    solver=HungarianSolver()
)
```

## Graph Directedness

**Directed vs Undirected Graphs**:
- **Directed (default)**: Each node has exactly k outgoing edges
  - Fast construction
  - Preserves asymmetric neighborhood structure
  - ⚠️ Use only with `:euclidean` or `:normalized` metrics

- **Undirected (`directed=false`)**: Edges are bidirectional (i↔j)
  - Nodes may have varying degrees (typically 1.5-2x the original k)
  - Required for geodesic metrics on manifolds
  - Original k preserved in `graph.metadata.original_k`

**Best Practice**: Always use `directed=false` with geodesic metrics (`:geodesic_*`) for manifold learning. Geodesic distances are inherently bidirectional on manifolds.

## Optimal Transport Solvers

All solvers use established libraries - no custom implementations.

### Exact OT Solvers (Recommended)

- **HungarianSolver** – O(k³), best for uniform distributions (via Hungarian.jl)
- **NetworkSimplexSolver** – Exact OT via network flow (OptimalTransport.jl + Tulip)
- **LPReferenceSolver** – General LP solver (JuMP + HiGHS)

### Approximate Solvers

- **SinkhornSolver** – Entropy-regularized OT (OptimalTransport.jl)
  - ⚠️ **Warning**: Default `reg=0.01` may be too small
  - Rule of thumb: `reg ≈ 5-10% of mean(cost_matrix)`
  - Returns approximate OT, not exact
  - Prefer NetworkSimplexSolver for production

- **GreedySolver** – Fast heuristic (O(k² log k))

**Recommendation**: Use `HungarianSolver()` with `NetworkSimplexSolver()` fallback.

## Distance Metrics

The `compute_all_curvatures` function supports flexible distance metrics:

### Available Metrics

- **`:euclidean`** – Direct Euclidean distance (default)
  - Works with directed or undirected graphs
  - Fast, no shortest path computation

- **`:geodesic_euclidean`** – Shortest path with Euclidean edge weights
  - ⚠️ Requires undirected graph for manifold learning
  - O(n³) pre-computation

- **`:geodesic_normalized`** – Shortest path with effective_epsilon weights (orcml)
  - ⚠️ Requires undirected graph for manifold learning
  - O(n³) pre-computation

- **`:geodesic_unit`** – Shortest path with unit weights (hop count)
  - ⚠️ Requires undirected graph for manifold learning
  - O(n³) pre-computation

- **`:normalized`** – effective_epsilon only
  - Works with directed or undirected graphs
  - Fast, no shortest path computation

### Configuration Examples

**Default (Fast)**:
```julia
curvatures = compute_all_curvatures(graph, data)
# Euclidean distances, directed graph
```

**Research Replication (orcml - ICLR 2025)**:
```julia
# Build undirected graph
graph = build_knn_graph(index, data; k=15, directed=false)

# Compute with orcml configuration
curvatures = compute_all_curvatures(
    graph, data;
    exclude_edge_endpoints=true,
    cost_metric=:geodesic_normalized,
    denominator_metric=:normalized,
    solver=HungarianSolver(),
    fallback_solver=NetworkSimplexSolver()
)
```

**Hybrid Geodesic**:
```julia
# Build undirected graph for geodesic metric
graph = build_knn_graph(index, data; k=15, directed=false)

curvatures = compute_all_curvatures(
    graph, data;
    cost_metric=:geodesic_euclidean,
    denominator_metric=:euclidean
)
```

## Performance

From benchmarks on graphs with n=1000, k=15, d=50:

| Config | Solver | ms/edge | Notes |
|--------|--------|---------|-------|
| Default | Hungarian | 0.073 | Fastest, uniform distributions |
| Default | NetworkSimplex | 0.517 | Robust fallback |
| orcml | LPReference | 2.269 | Best for non-uniform |
| orcml | NetworkSimplex | 2.853 | Close second |

**Key findings:**
- Hungarian is 30x faster for uniform distributions
- orcml config is ~30x slower (undirected + geodesic)
- SinkhornSolver had 100% failure rate with default parameters

⚠️ **Note**: Geodesic distance metrics (`:geodesic_*`) require O(n³) Floyd-Warshall pre-computation for shortest paths. This is computed once per `compute_all_curvatures()` call, then O(1) lookups. For n=1000: ~1B operations but acceptable for offline analysis.

See `docs/references/ORC_PERFORMANCE_ANALYSIS.md` for detailed benchmarks.

## Graph Pruning Workflows

### Graph-Based Indices (NNDescentIndex)

```julia
# 1. Build index
index = build_index(NNDescentIndex, data; k=20)

# 2. Extract internal graph
graph = index.graph

# 3. Compute curvatures
curvatures = compute_all_curvatures(graph, data)

# 4. Filter
filtered_graph = filter_graph(graph, data; curvatures=curvatures)

# 5. Rebuild with refined graph
refined_index = build_index(NNDescentIndex, data;
    k=filtered_graph.k,
    initial_graph=filtered_graph
)
```

### Non-Graph Indices (HNSW, KDTree, LSH)

```julia
# 1. Build index
index = build_index(HNSWIndex, data; M=16)

# 2. Extract k-NN graph
graph = build_knn_graph(index, data; k=20)

# 3. Compute curvatures
curvatures = compute_all_curvatures(graph, data)

# 4. Filter
filtered_graph = filter_graph(graph, data; curvatures=curvatures)

# 5. Build new index from refined graph
refined_index = build_index(NNDescentIndex, data;
    k=filtered_graph.k,
    initial_graph=filtered_graph
)
```

**Note**: Non-graph indices cannot be modified in-place. Best approach is to build a NNDescentIndex from the refined graph.

## Examples

Complete working examples:

- `docs/examples/graphs/05-curvature-filtering.jl` – Basic filtering workflow
- `docs/examples/graphs/08-curvature-pruning-graph-index.jl` – Graph-based index pruning
- `docs/examples/graphs/09-curvature-pruning-non-graph-index.jl` – Non-graph index pruning
- `docs/examples/graphs/10-orc-technique-comparison.jl` – Compare different ORC approaches

## Technical Details

### effective_epsilon

In orcml configuration, `effective_epsilon` provides scale normalization:

For each edge (x, y):
```
eps_x = mean(k-nearest neighbor distances from x)
eps_y = mean(k-nearest neighbor distances from y)
effective_eps(x, y) = max(eps_x, eps_y)
```

This value is used as:
1. Edge weight in shortest path computation
2. Denominator in curvature formula: κ = 1 - W₁/effective_eps

**Key insight**: Each edge gets its own normalization value based on the coarser scale of its two endpoints.

See `docs/references/EFFECTIVE_EPSILON_EXPLAINED.md` for detailed explanation.

### Sinkhorn Regularization

The Sinkhorn solver requires careful tuning:

```julia
# Estimate appropriate regularization from data
data_scale = mean(norm(data[:, i] - data[:, j])
                  for i in 1:min(100,n), j in i+1:min(100,n))
solver = SinkhornSolver(reg=0.1 * data_scale, atol=1e-6)
```

See `docs/references/SINKHORN_DIAGNOSIS.md` and `scripts/diagnose_sinkhorn.jl` for diagnostics.

## References

- **Paper**: "Recovering Manifold Structure Using Ollivier-Ricci Curvature" (ICLR 2025)
- **orcml package**: https://github.com/TristanSaidi/orcml
- **Performance analysis**: `docs/references/ORC_PERFORMANCE_ANALYSIS.md`
- **Benchmarks**: `benchmark_results/orc_benchmark_julia.csv`
- **Test scripts**: `scripts/test_orcml_exact_match.jl` (99.65% correlation with orcml.py)
