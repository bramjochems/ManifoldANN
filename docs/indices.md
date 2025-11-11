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

## Common Guidance
- Indices never store the raw dataset; always pass `data` to `query`.
- Use `supports_mutation(index)` to branch on whether `insert!` is available.
- Prefer `BruteForceIndex` when validating new algorithms to keep test coverage high and to quantify recall for probabilistic structures.
