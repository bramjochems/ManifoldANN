# ADR-0007: HNSW Index Architecture

## Context

We need a high-recall approximate index that also serves as the foundation for
future graph workflows (layer manipulation, custom topologies). HNSW fits this
role but introduces algorithmic complexity (hierarchical layers, greedy
traversal, local rewiring). Capturing the design decisions up front keeps the
implementation transparent while allowing future tweaks (custom planners,
policies, or hand-crafted layers).

## Decision

1. **Metadata-Only Graph**: `HNSWIndex` stores only adjacency lists per layer,
   entry point, max layer, and tuning parameters. The actual data matrix remains
   external (consistent with ADR-0002). Inserts require callers to append the
   new column to their own storage before invoking `insert!`.

2. **Pluggable Components**:
   - *Layer planner*: defaults to an exponential sampler but is abstracted via
     `AbstractLayerPlanner` so future experiments (fixed layers, manual seeds)
     can override it.
   - *Neighbor policy*: `AbstractNeighborPolicy` governs how candidate pools are
     pruned (default: keep the closest `M`). This hook will allow diversity- or
     tangent-aware selections later.
   - *Traversal policy*: `AbstractTraversalPolicy` encapsulates greedy descent +
     ef-search logic. Right now only `GreedyTraversalPolicy` exists, but the
     structure isolates the hot loop for future multi-probe variants.

3. **Explicit Parameters**: HNSW-specific knobs (`M`, `ef_construction`,
   `ef_search`) are surfaced as builder/query keywords. `ef_search` defaults to
   the traversal policy’s setting but can be overridden per query, matching
   standard HNSW implementations.

4. **Graph Integration**: Because HNSW already maintains adjacency lists, it
   plugs directly into `build_knn_graph` without duplicating storage, enabling
   graph workflows to reuse the approximate structure.

## Consequences

- The implementation mirrors the textbook HNSW algorithm but keeps each phase
  (layer planning, traversal, neighbor selection) modular, so custom behavior
  can be inserted later without rewriting the index.
- Mutation is supported (`supports_mutation` returns true) while keeping the
  “indices never store raw data” rule intact.
- Tests and examples can validate both the reference behavior (matching brute
  force on small datasets) and mutation flows (insert new points, rebuild
  graphs) without hand-wiring new scaffolding every time.
