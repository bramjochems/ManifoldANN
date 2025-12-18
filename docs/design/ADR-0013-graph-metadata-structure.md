# ADR-0013: Graph Metadata Structure Enhancement

## Context

ADR-0006 introduced optional per-node metadata via `KNNGraph.metadata` as a
simple vector or `nothing`. This worked well for attaching labels or coordinates.

However, new ORC features require storing **structural metadata** about the graph:
1. **original_k**: The k requested at construction (may differ from `graph.k` for undirected graphs)
2. **directed**: Whether the graph is directed or symmetrized (critical for geodesic metrics)

These are graph-level properties, not per-node data. Storing them separately
would fragment the API and complicate backward compatibility.

## Decision

1. **Hierarchical Metadata Structure**:
   ```julia
   # Before (ADR-0006):
   graph.metadata = ["label1", "label2", ...] | nothing

   # After (ADR-0013):
   graph.metadata = (
       original_k = 15,             # Structural: k requested at construction
       directed = false,            # Structural: is graph directed?
       node_metadata = ["label1", ...]  # Per-node: user data
   )
   ```

2. **Accessor Functions**:
   - `graph_metadata(graph)`: Returns entire metadata tuple
   - `node_metadata(graph, i)`: Returns metadata for node i (same API as before)
   - `has_metadata(graph)`: Returns true if node_metadata is present

3. **Structural Metadata Access**:
   ```julia
   # Direct field access (for internal use)
   graph.metadata.original_k
   graph.metadata.directed

   # Helper function (for external use)
   _get_original_k(graph)  # Handles missing metadata gracefully
   ```

4. **Construction**:
   ```julia
   _create_graph_metadata(original_k, directed, node_metadata, n_points)
   ```
   - Centralizes metadata creation logic
   - Ensures consistent structure across codebase
   - Handles `nothing` → `nothing` for node_metadata

## Consequences

**Benefits**:
- **Single source of truth**: All metadata in one place
- **API compatibility**: `node_metadata(graph, i)` still works
- **Type stability**: Metadata always has the same structure
- **Introspection**: Code can query if graph is directed/undirected

**Breaking Changes**:
- Direct access `graph.metadata[i]` breaks (must use `node_metadata(graph, i)`)
- `graph_metadata(graph)` now returns named tuple, not vector
- Tests updated to use new accessors

**Migration**:
```julia
# Before:
labels = graph_metadata(graph)
label_i = graph.metadata[i]

# After:
labels = graph_metadata(graph).node_metadata
label_i = node_metadata(graph, i)
```

**Use Cases**:

```julia
# effective_epsilon needs original_k for undirected graphs
k = _get_original_k(graph)  # Gets metadata.original_k if available

# Geodesic metrics need to know if graph is directed
if graph.metadata.directed
    @warn "Geodesic metrics work best on undirected graphs"
end
```

## Rationale

**Why not separate fields?**
- Would add `KNNGraph.original_k` and `KNNGraph.directed` fields
- Breaks existing constructors and serialization
- Metadata is the natural extension point (established in ADR-0006)

**Why named tuple instead of struct?**
- Lightweight (no new types)
- Easy to extend with additional fields
- Backward compatible with `nothing` for node_metadata

**Why store original_k instead of just using graph.k?**
- For undirected graphs, `graph.k` is max degree after symmetrization
- Original k is needed for:
  - effective_epsilon computation (select k-nearest from degree > k)
  - Algorithm documentation ("built with k=15")
  - Reproducibility (what was requested vs what resulted)

## Future Work

- Consider adding `created_at` timestamp
- Add `algorithm` field (which index was used)
- Store distance metric used for construction
- Explore serialization/deserialization helpers

## References

- ADR-0006: Graph Metadata Support (per-node data)
- ADR-0012: Flexible Distance Metrics (uses directed field)
- ADR-0014: Undirected Graph Support (uses original_k field)
