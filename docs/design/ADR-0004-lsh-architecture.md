# ADR-0004: LSH Architecture and Randomized Index Patterns


## Context

Locality-sensitive hashing (LSH) introduces new architectural constraints:
multiple hash families targeting different metrics, randomized projections,
and optional mutation (insert) support even though indices never own the raw
data. While the immediate motivation is LSH, the resulting patterns also apply
to other randomized indices (e.g. RP trees, HNSW variants). Capturing these
decisions prevents ad-hoc duplication as additional ANN structures arrive.

## Decision

1. **Hash-Family Abstraction**: LSH hash functions conform to the
   `AbstractLSHHash` interface (`hash_point`, `hash_batch`, `distance_function`).
   Factories such as `make_random_hyperplane_hash` or `make_binning_hash`
   produce concrete hashes with typed storage (projection matrices, offsets).
   `LSHIndex` remains generic over `H<:AbstractLSHHash`, letting us plug in
   angular, L2, or future families (e.g. spectral hashes) without touching the
   table/index implementation. This mirrors the general pattern we want for
   other index families: define slim traits so algorithmic variants remain
   composable.

2. **Data Ownership**: Consistent with ADR-0002, `LSHIndex` never stores the raw
   dataset. Builders record only metadata (dimension, point count, hash tables);
   callers provide `data` at query time. Inserts (`insert!`) merely update the
   tracked point count and append ids to buckets; the caller remains responsible
   for appending columns to their matrix before invoking `query`. This keeps
   memory usage predictable and lets us reuse indices across denoised/filtered
   representations.

3. **Deterministic RNG Strategy**: Randomized builders must produce reproducible
   structures when given the same RNG seed. `LSHIndex` achieves this by spawning
   independent RNGs per table (`_spawn_rngs`) instead of sharing one global
   generator. The same approach should be reused anywhere we need many random
   subcomponents (e.g. HNSW layers, projection forests) so test fixtures and
   examples remain stable.

4. **Vectorized Hashing & Batch Utilities**: Hash tables are built via bulk
   matrix multiplies (`hash_batch`) before packing signatures into `UInt64`
   keys. This keeps Julia on BLAS paths, minimizes allocations, and makes batch
   insertions straightforward. Future randomized indices should follow the same
   pattern: provide batch-oriented helpers so builders and mutators can stay
   efficient and type-stable.

5. **Mutation Policy**: LSH supports `insert!` by updating metadata only. The
   general rule stands: indices may opt into mutation via `supports_mutation`
   and must document which arguments keep metadata consistent. If rebuilds are
   required (e.g. to rebalance tables), that remains the caller’s choice; the
   index never issues implicit rebuilds.

## Consequences

- New hash families can be added by implementing `AbstractLSHHash` plus a
  factory, without altering `LSHIndex`.
- Deterministic RNG splitting becomes the default expectation for any module
  with randomized state, simplifying tests and reproducible examples.
- Batch helpers encourage performant builders/inserts in other indices as well.
- Documentation/examples highlight that data ownership always lives outside the
  index, reinforcing ADR-0002’s guidance.
