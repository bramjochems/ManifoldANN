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
- [ ] Retain neighbor distances through construction to enable a diversification/pruning pass similar to PyNNDescent’s angular check.
- [ ] Add an optional RP-tree or projection-tree seeding phase before random initialization to shrink required iterations.
- [ ] Experiment with incremental symmetry (apply symmetry every few iterations) so reverse edges help convergence earlier.
- [ ] Explore `Threads.@threads` or a modulo-based partitioning scheme to parallelize candidate evaluation once data structures are thread-safe.
