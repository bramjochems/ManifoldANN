# ADR-0014: Undirected Graph Support for ORC Computation

## Context

The original `build_knn_graph()` created directed graphs where node i has k
outgoing edges to its k nearest neighbors. This is standard for ANN applications
(routing, navigation, index construction).

Ollivier-Ricci curvature (ORC) can be computed on both directed and undirected
graphs, but they have different interpretations:
- **Directed**: Asymmetric neighborhoods, measures directional flow/curvature
- **Undirected**: Symmetric neighborhoods, measures bidirectional local geometry

The orcml paper (ICLR 2025) uses **undirected k-NN graphs** in their approach.
Their implementation symmetrizes the k-NN graph to ensure:
- Geodesic distances are symmetric: d(i,j) = d(j,i)
- Neighborhood structure is consistent (if i∈N(j), then j∈N(i))
- Curvature reflects bidirectional local geometry

To replicate orcml and support research comparing directed vs undirected ORC,
we need undirected graph support.

## Decision

1. **Add `directed` Parameter**:
   ```julia
   build_knn_graph(index, data; k=15, directed=true)  # Existing behavior
   build_knn_graph(index, data; k=15, directed=false) # New: symmetrize
   ```

2. **Symmetrization Algorithm**:
   ```julia
   function _symmetrize_adjacency(adjacency, n_points)
       # For each edge i→j, add reverse edge j→i
       # Use sets for efficient union, then convert to sorted vectors
   ```
   - Takes union of i→j and j→i edges
   - Results in varying degrees (typically 1.5-2x original k)
   - Edges are sorted for consistent iteration order

3. **K-Value Semantics**:
   ```julia
   # Directed graph: graph.k == k (requested)
   # Undirected graph: graph.k == max_degree (after symmetrization)
   #                   graph.metadata.original_k == k (requested)
   ```
   - **graph.k**: Maximum degree across all nodes (for memory allocation)
   - **metadata.original_k**: The k requested at construction (for algorithms)

4. **Why Use original_k?**
   - effective_epsilon selects k-nearest from potentially >k neighbors
   - Documentation clarity: "built with k=15" means original_k=15
   - Algorithm reproducibility: same k across directed/undirected

5. **Validation for Geodesic Metrics**:
   ```julia
   validate_geodesic_config(graph, metric)
   # Warns if using geodesic metrics on directed graphs
   # Note: Not an error - both are valid, but have different interpretations
   ```

## Consequences

**Benefits**:
- **Research flexibility**: Compare directed vs undirected ORC formulations
- **orcml replication**: Exact match with reference implementation
- **Symmetric geodesics**: Undirected graphs ensure d(i,j) = d(j,i)
- **Bidirectional geometry**: Natural for applications assuming mutual proximity
- **User choice**: Directed for asymmetric flow, undirected for symmetric structure

**Semantic Complexity**:
- **graph.k changes meaning** for undirected graphs:
  - Directed: k = requested neighbors
  - Undirected: k = max degree after symmetrization
- **Requires documentation**: Users must understand k vs original_k

**Performance**:
- Symmetrization: O(n·k) using sets
- Memory: Undirected graphs have more edges (~1.5-2x)
- Algorithm cost: More edges → more curvature computations

**Example Usage**:

```julia
# Directed graph (default - asymmetric neighborhoods)
index = build_index(BruteForceIndex, data)
graph_directed = build_knn_graph(index, data; k=15, directed=true)

# Undirected graph (symmetric neighborhoods)
graph_undirected = build_knn_graph(index, data; k=15, directed=false)

# orcml-style curvature (uses undirected graphs)
curvatures = compute_all_curvatures(graph_undirected, data;
    cost_metric=:geodesic_normalized,
    denominator_metric=:normalized,
    exclude_edge_endpoints=true
)

# Filter low-curvature edges (works with both directed and undirected)
filtered = filter_graph(graph_undirected, data;
    curvature_threshold=0.0,
    solver=HungarianSolver()
)
```

**Edge Cases**:
- Self-loops: Always included (even when x == y and x < y fails)
- Fixed in `_collect_edges_to_process()` with explicit self-loop handling

## Alternatives Considered

**Option 1: Separate Constructor**
```julia
build_symmetric_knn_graph(index, data; k)
```
- Pro: Clearer API, no semantic confusion about k
- Con: Code duplication, more functions to maintain
- Rejected: `directed` parameter is simpler

**Option 2: Always Store original_k**
```julia
struct KNNGraph
    k::Int            # Actual max degree
    k_original::Int   # Requested k
end
```
- Pro: Explicit, no metadata access needed
- Con: Breaks existing constructors, more struct fields
- Rejected: Metadata extension is less invasive (ADR-0013)

**Option 3: Keep graph.k == original_k, add max_degree field**
```julia
graph.k == 15  # Always requested k
graph.max_degree == 28  # Actual max after symmetrization
```
- Pro: k semantics unchanged
- Con: Almost all code uses max_degree, not k
- Rejected: graph.k is used for memory allocation

## Future Work

- Lazy symmetrization (compute edges on-demand)
- Weighted undirected graphs (preserve distances in edges)
- Degree regularization (limit max degree after symmetrization)
- Profile memory usage (undirected graphs are larger)

## References

- "Recovering Manifold Structure Using Ollivier-Ricci Curvature" (ICLR 2025)
- orcml: https://github.com/TristanSaidi/orcml (uses undirected graphs)
- ADR-0012: Flexible Distance Metrics (geodesic metrics need undirected)
- ADR-0013: Graph Metadata Structure (stores original_k and directed)
