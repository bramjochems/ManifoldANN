# Indices Overview

This package exposes several ANN backends. They all create structures called an AbstractANNIndex, which is the object used for searching for (approximate) nearest neighbours. All structures have in common that they can be created through a common ```build_index``` interface and queried through a ```query``` interface. All of these are defined in ```src/ann_index.jl```.

Below is a quick reference of the currently implemented indices and their most relevant options.

## BruteForceIndex
- **Scope**: correctness baseline, tiny datasets, or ground-truth evaluations.
- **Key options**: `query` accepts `distance=default_distance` so users can swap in custom metrics. `insert!` is supported because the index only tracks the point count.
- **Usage tips**: Pair with `recall_at_k` to benchmark approximate methods or to verify examples (see `docs/examples/indices/01-...jl`).

## KDTreeIndex
- **Scope**: exact search for low/medium dimensional Euclidean spaces.
- **Builder keywords**:
  - `axis_selector`: `:variance` (default) picks the axis with largest spread in each subtree; `:cyclic` cycles through axes deterministically.
- **Query keywords**:
  - `distance`: defaults to Euclidean; any metric that obeys hyperplane pruning can be supplied (otherwise both subtrees are explored).
- **Mutation**: immutable—rebuild when the dataset changes materially.
- **Examples**: `docs/examples/indices/ex03-kdtree-index.jl` compares variance vs cyclic policies.

## LSHIndex
- **Scope**: approximate search with locality-sensitive hashing families.
- **Builder keywords** (see ADR-0004 for rationale):
  - `n_tables`, `hash_length`: control recall/latency trade-offs.
  - `hash_factory`: e.g. `make_random_hyperplane_hash` (angular) or `make_binning_hash` (L2). Additional kwargs like `bin_width` and `use_offset` are forwarded to the factory.
  - `rng`: ensures reproducible tables; per-table RNGs are spawned under the hood.
- **Query keywords**:
  - `candidate_cap`: optional clamp on unique candidates before distance refinement.
- **Mutation**: `insert!` updates buckets and the point count; callers still own the data matrix.
- **Examples**: `docs/examples/indices/ex02-lsh-index.jl` shows both angular and binning hashes along with recall reporting vs brute force.

## HNSWIndex
- **Scope**: graph-based approximate search with hierarchical layers; supports mutation (`insert!`) and layer-aware experimentation.
- **Builder keywords**:
  - `M`: max degree per layer (default 16).
  - `ef_construction`: controls candidate breadth during insertion; higher improves recall.
  - `ef_search`: default breadth during queries (can be overridden per call).
  - `planner`, `neighbor_policy`, `traversal_policy`: optional hooks for custom layer assignment, connection heuristics, or traversal strategies.
- **Query keywords**:
  - `ef_search`: optional override (automatically clamped to `>= k`).
- **Mutation**: `insert!(index, data, point; rng=...)` expects the caller to append the new column to `data` first; the index updates only its graph metadata.
- **Examples**: `docs/examples/indices/ex04-hnsw-index.jl` (index usage) and `docs/examples/graphs/04-hnsw-knn-graph.jl` (graph export).

## MultiLevelIndex
- **Scope**: FAISS-like hierarchical indices with configurable transforms, routing, and merge strategies; enables IVF, multi-level clustering, and future PQ/OPQ pipelines.
- **Builder keywords**:
  - `config`: Declarative configuration tree specifying index structure (see below)
  - `merge_strategy`: Strategy for combining results from multiple child indices (default: `SimpleMerge()`)
- **Configuration**: Build using nested `TransformedConfig` and `TerminalConfig`:
  ```julia
  # IVF: KMeans(100) → HNSW per cluster
  config = TransformedConfig(
      KMeansTransform(k=100, distance=Euclidean(), init=:kmeans_plus_plus),
      TopKRouting(5),  # Probe 5 nearest clusters
      TerminalConfig(HNSWIndex, (M=16, ef_construction=200))
  )
  index = build_index(MultiLevelIndex, data, config)
  ```
  - Convenience: `build_ivf_hnsw_index(data; nlist=100, routing_k=8, hnsw_M=16, ...)` wires this pattern up directly.
- **Transforms**:
  - `IdentityTransform()`: Pass-through (useful for testing or mixed hierarchies)
  - `KMeansTransform(k, distance; init=:kmeans_plus_plus)`: Cluster-based partitioning
    - `init` can be `:random` or `:kmeans_plus_plus` (default)
  - Future: `PQTransform`, `OPQTransform`, `PCATransform`
- **Routing Strategies**:
  - `TopKRouting(k)`: Probe k nearest clusters/partitions
  - `ExhaustiveRouting()`: Probe all children (useful with IdentityTransform)
- **Merge Strategies**:
  - `SimpleMerge()`: Trust sub-index distances, sort globally, deduplicate
  - Future: `RecomputeMerge` for accuracy with approximate distance computations
- **Query keywords**: None currently; routing/merge behavior is configured at build time
- **Mutation**: Immutable—rebuild when dataset changes
- **Examples**:
  - `docs/examples/indices/06-ivf-index.jl` (basic IVF)
  - `docs/examples/indices/07-multilevel-hierarchy.jl` (two-level clustering)
  - See `docs/transforms.md` for detailed transform documentation

## Common Guidance
- Indices never store the raw dataset; always pass `data` to `query`.
- Use `supports_mutation(index)` to branch on whether `insert!` is available.
- Prefer `BruteForceIndex` when validating new algorithms to keep test coverage high and to quantify recall for probabilistic structures.
