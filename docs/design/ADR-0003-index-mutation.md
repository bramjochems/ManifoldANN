# ADR-0003: Index Mutation Hooks


## Context

Some ANN structures (e.g. brute-force scans, potential graph-based indices) can
accept new points after initial construction without rebuilding the entire
index. Other indices remain immutable for predictability. We need a uniform way
to advertise mutation capabilities and a default behavior for indices that do
not support updates.

## Decision

- `AbstractANNIndex` exposes the predicate `supports_mutation(index)` (default:
  `false`) and a generic `insert!(index, args...; kwargs...)` function.
- Mutable indices override both the predicate and `insert!`, documenting the
  accepted arguments (typically the new point or batch). Since indices do not
  own the dataset, `insert!` only updates structural metadata; callers remain
  responsible for adding the new point to their data storage.
- Immutable indices inherit the default `insert!` method, which throws an
  informative `ArgumentError`.
- `BruteForceIndex` serves as the first example of a mutable index: it tracks
  dataset dimensionality and a point count, validates inserts, and increments
  the count while leaving the data ownership to the caller.

## Consequences

- Downstream code can branch on `supports_mutation(index)` before attempting
  dynamic updates.
- Mutable indices must be thoroughly tested (unit + property tests) to ensure
  inserts maintain invariants and metadata consistency.
- Immutable indices incur no additional complexity because the default method
  handles errors uniformly.
