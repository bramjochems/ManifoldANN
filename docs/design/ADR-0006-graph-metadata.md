# ADR-0006: Graph Metadata Support

## Context

Upcoming workflows (manifold-aware shortest paths, tangent propagation,
denoising diagnostics) require attaching per-node metadata to kNN graphs. Until
now `KNNGraph` stored only neighbor ids, `k`, and `include_self`, making it
impossible to ship auxiliary payloads alongside graph construction. We need a
lightweight extension that keeps existing indices untouched while allowing
callers to opt into metadata.

## Decision

1. **Graph-Level Metadata Field**: `KNNGraph` now carries an optional
   `metadata::Union{Nothing,Vector{T}}`. `build_knn_graph` accepts a
   `metadata=` keyword; when provided, the vector is copied (to keep types
   concrete) and validated to match the dataset length.

2. **Accessors and Traits**: Utility functions
   `has_metadata(graph)`, `graph_metadata(graph)`, and
   `node_metadata(graph, i)` provide ergonomic access and input validation,
   mirroring the capability introspection already present for indices.

3. **Non-Intrusive Default**: Passing no metadata keeps the field `nothing`,
   preserving the memory footprint and behavior of existing code. Examples and
   docs highlight metadata as an optional enhancement rather than a required
   field.

## Consequences

- Downstream algorithms can ship typed payloads (tangent bases, PCA coords,
  labels) together with the graph without inventing parallel data structures.
- Tests now cover both metadata and non-metadata cases, ensuring regressions
  (e.g. mismatched lengths) surface early.
- Documentation (`docs/graphs.md` and graph examples) explicitly explains how
  to attach and consume metadata, guiding contributors as more graph-aware
  features are added.
