# ADR-0005: KD-Tree Index Design

## Context

We need an exact-yet-fast spatial index to complement `BruteForceIndex` and the
randomized LSH family. KD-trees provide logarithmic search time for
low/medium-dimensional Euclidean data, but they come with design trade-offs:
balancing strategies, data ownership (per ADR-0002 indices must not store the
raw matrix), immutability, and axis-selection heuristics. Before adding the
implementation we codify how these aspects behave so future hierarchical trees
(RP trees, spill trees, etc.) stay consistent.

## Decision

1. **Metadata-Only Storage**: `KDTreeIndex` stores only structural metadata
   (split axis, split threshold, child references, and the id of the point at
   each node). It never owns raw coordinates. Queries always receive the data
   matrix, keeping memory usage predictable and enabling callers to reuse the
   same tree with denoised/projected copies.

2. **Axis Selection**: Builders default to picking the axis with the largest
   spread inside each subtree (variance proxy). A deterministic `:cyclic`
   policy is also available for reproducibility experiments. This matches the
   lightweight, easy-to-reason-about style described in ADR-0001 while still
   producing balanced trees for typical datasets.

3. **Immutability**: KD-trees do not expose `insert!`. Maintaining balance
   after arbitrary inserts is expensive and would require per-node bounding
   boxes or rebuilds. Instead, callers rebuild when data drift warrants it.
   This aligns with ADR-0003’s guidance: indices must explicitly opt into
   mutation and document how consistency is preserved.

4. **Euclidean Search Contract**: The pruning logic assumes
   Euclidean-compatible distances (the default `default_distance`). We surface
   this expectation in docstrings so future metrics either adapt the bounds or
   fall back to brute-force scans.

## Consequences

- KD-trees offer an exact baseline that is dramatically faster than brute
  force on suitably low-dimensional data while adhering to the “no raw data
  stored” rule.
- Insertions require rebuilding, keeping the implementation small and easy to
  audit; if incremental updates become necessary we can explore buffered
  rebuild strategies in a follow-up ADR.
- Axis-selection policies are explicit and extensible, serving as a template
  for additional tree-based indices (spill trees, cover trees) that may join
  later.
