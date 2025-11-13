# TODO

## Performance Improvements for HNSW Index Building

**Context:** Current HNSW build time is ~5× slower than hnswlib/FAISS on my laptop(4s vs 0.7s for 10K points). This is acceptable for a research codebase prioritizing flexibility, but there are optimization opportunities that preserve the experimental design.

### Key Differences vs hnswlib/FAISS

**1. Memory Layout (30-40% speedup potential)**
- **Current:** Nested `Vector{Vector{Vector{Int}}}` structure causes cache misses and GC pressure
- **hnswlib/FAISS:** Flat contiguous arrays with offset indexing (`neighbors[offsets[node_id]]`)
- **Trade-off:** Medium implementation complexity, LOW flexibility impact

**2. Neighbor Selection Heuristic (implemented, needs tuning)**
- **Current:** `DiversifiedNeighborPolicy` is now the default, matching FAISS's connectivity-aware pruning; `HeuristicNeighborPolicy` remains available for ablation comparisons
- **hnswlib/FAISS:** Connectivity-aware pruning (reject dominated neighbors)
- **Status:** Feature exists; follow-up work is benchmarking to quantify the recall/build-time trade-offs and documenting when to opt into the heuristic variant

**3. SIMD Distance Functions (10-20% speedup potential)**
- **Current:** Generic distance function with runtime polymorphism
- **hnswlib/FAISS:** Hand-coded SSE/AVX intrinsics
- **Trade-off:** HIGH flexibility impact - **NOT RECOMMENDED** (defeats thesis goal of metric flexibility)


**4. Parallel Construction (4-8× speedup potential)**
- **Current:** Sequential insertion
- **hnswlib:** Lock-striped parallel insertion (65K lock bins)
- **Trade-off:** Very high complexity, non-deterministic graphs - **NOT RECOMMENDED** for research


### Recommended Future Work

**Priority 1 (Medium effort, high value):**
- [ ] Implement flat memory layout for adjacency lists
  - Replace `Vector{Vector{Vector{Int}}}` with flat storage + offsets
  - Expected: 30-40% build time reduction
  - Preserves all flexibility for testing neighbor policies

**Not Recommended:**
- ❌ SIMD-specialized distances (loses metric flexibility)
- ❌ Parallel construction (complexity not worth it for research use case)

## NN-Descent Performance Improvements

**Context:** The bounded candidate pairing reduced per-iteration work, but PyNNDescent still outperforms us thanks to higher-quality initialization, diversification, and parallel update machinery.

### Potential Follow-ups

- [ ] Tune `max_candidate_neighbors` defaults per dataset/metric and document recommended settings.
- [ ] Retain neighbor distances through construction to enable a diversification/pruning pass similar to PyNNDescent's angular check.
- [ ] Add an optional RP-tree or projection-tree seeding phase before random initialization to shrink required iterations.
- [ ] Experiment with incremental symmetry (apply symmetry every few iterations) so reverse edges help convergence earlier.
- [ ] Explore `Threads.@threads` or a modulo-based partitioning scheme to parallelize candidate evaluation once data structures are thread-safe.

## Multi-Level Index Optimizations

**Context:** The multi-level index architecture (IVF, multi-level hierarchies) is now implemented with two key optimizations already in place: parallel child index building and fast-path IdentityTransform. Additional batch transform optimizations are deferred until profiling justifies the complexity.

### Implemented Optimizations

- [x] **Parallel child index building** (`src/indices/multilevel/builder.jl`)
  - Child indices are built in parallel using `Threads.@threads`
  - Near-linear speedup potential (e.g., 4× with 4 threads)
  - Biggest win for IVF with many clusters or deep hierarchies

- [x] **Fast-path IdentityTransform** (`src/transforms/utils.jl`)
  - Specialized `apply_transform_batch(::IdentityTransform, X)` returns input directly
  - Zero-copy optimization for pass-through transforms
  - Eliminates allocation overhead in non-bucketing hierarchies

### Deferred Optimizations

**Batch Transform API** (defer until profiling shows bottleneck):

- [ ] Add `transform_batch(transform::AbstractTransform, X::Matrix)` interface
  - **Rationale:** Enable transforms to provide vectorized implementations
  - **Example:** KMeans could compute `pairwise_distances!(X, centroids)` once instead of per-column
  - **Complexity:** Requires handling heterogeneous `TransformResult` types
  - **Current status:** Per-column overhead likely small compared to child index building

- [ ] Implement `transform_batch` for KMeansTransform
  - Compute all point-to-centroid distances in single pass
  - Eliminates `n` separate `compute_distances` calls in `partition_by_transform`
  - Expected speedup: 10-30% for transforms with expensive distance computations

- [ ] Extend `apply_transform_batch` to use optional `transform_batch`
  - Fall back to per-column loop if batch method not available
  - Maintains backward compatibility with existing transforms

**When to revisit:**
- If profiling shows `partition_by_transform` is >20% of build time
- When implementing PQ/OPQ transforms that would benefit from batch processing
- If datasets grow large enough that allocation overhead dominates
