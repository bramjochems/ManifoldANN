# ADR-0008: NN-Descent Graph Index

## Context

HNSW covers navigable small-world graphs, but we also want a batch-oriented,
layer-free graph builder that can act as both an ANN index and an explicit
`k`-NN graph constructor. NN-Descent provides a deterministic single-layer
graph whose quality can be traded against build time via sampling and
iteration limits. We need an implementation that honors the shared
`AbstractANNIndex` interface, keeps distances flexible, and leaves room to
swap out sampling or neighbor-update heuristics later.

## Decision

1. Introduce `NNDescentIndex <: AbstractGraphIndex` that stores only the
   neighbor lists (no raw data) and exposes `materialize_graph`.
2. Model algorithm knobs as explicit fields on the index:
   - `k`, `max_iterations`, `convergence_threshold`
   - `distance` functor (defaults to `default_squared_distance`)
   - `sampling_policy::AbstractNNDescentSamplingPolicy`
3. Provide a default `UniformPairSampling` policy plus a resolver so future
   strategies can be injected without widening method signatures. Builder
   logic calls `should_consider_pair(policy, rng)` for every candidate pair,
   making sampling the only policy-specific hook for now.
4. Keep construction in `src/indices/nndescent/builder.jl` (initialization,
   iterative refinement, helper structs) and query/memory helpers in
   `src/indices/nndescent/query.jl` to keep files short per ADR-0001.
5. Reuse shared utilities (`validate_index_dimensions`, `NeighborMinHeap`,
   `spawn_child_rngs`) so the NN-Descent search path stays type-stable and
   consistent with other indices.

## Consequences

- `NNDescentIndex` can plug into any workflow that expects an
  `AbstractGraphIndex`; `build_knn_graph` simply copies the stored adjacency.
- Policies remain pluggable: adding degree-aware sampling or alternative
  candidate scoring only requires new `AbstractNNDescentSamplingPolicy`
  implementations.
- Queries reuse the same benchmarking + evaluation harness as HNSW and
  benefit from the SIMD-friendly squared-distance helper introduced earlier.
- Tests and docs live alongside other indices (unit tests under
  `test/unit/indices/nndescent`, runnable example under
  `docs/examples/indices/05-nndescent-index.jl`), preserving coverage and
  discoverability expectations from AGENTS.md.
