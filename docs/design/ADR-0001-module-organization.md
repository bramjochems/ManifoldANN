# ADR-0001: Module Organization and Directory Layout

## Context

ManifoldANN must support multiple approximate nearest-neighbor (ANN) structures, explicit kNN graphs with custom node metadata (e.g. tangents, PCA projections), and downstream algorithms such as shortest paths. The source tree should stay readable with short files, type-specialized code, and room for future preprocessing and workflow layers.

## Decision

We organize `src/` as follows:

- `ManifoldANN.jl`: light top-level module that defines exports and `include`s.
- `ann_index.jl`: shared traits and interface (`AbstractANNIndex`, `AbstractGraphIndex`, `build_index`, `query`, capability introspection).
- `indices/`: ANN structures (brute force, KD trees, HNSW variants, NN-Descent, LSH, etc.). Each submodule owns construction, search policies, and variant-specific hooks while conforming to the shared interface. HNSW remains here even though it exposes graph layers.
- `graphs/`: canonical `KNNGraph` types, node payloads, metadata traits, graph materializers, and algorithms that inherently require an explicit graph (shortest paths can live here when implemented).
- `preprocessing/`: denoising, PCA, tangent estimation, or other transforms provided as pure functors so builders can compose them.
- `metrics/`: wrappers and adapters around `Distances.jl` plus manifold-aware metrics.
- `workflows/`: optional high-level orchestration to chain preprocessing, indexing, graph materialization, and downstream tasks (can be added later).

Other folders (e.g. `paths/`) will be introduced under `graphs/` once needed.

## Consequences

- ANN indices remain grouped, making it easy to compare implementations and share utilities.
- Graph-specific logic is isolated, so storing custom node metadata or running graph algorithms never pollutes lower-level search code.
- Adding new preprocessing or workflow steps does not disturb the core ANN or graph modules.
- The consistent layout keeps file lengths manageable and helps Julia specialize aggressively because related code is co-located and type parameters stay explicit.
