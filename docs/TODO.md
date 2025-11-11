# TODO

- [ ] Flesh out module structure for manifold-specific ANN utilities.
- [ ] Implement initial benchmarks.
- [ ] Document datasets and preprocessing steps.

## Performance Improvements for HNSW Index Building

**Context:** Current HNSW build time is ~5× slower than hnswlib/FAISS (15.6s vs 3s for 10K points). This is acceptable for a research codebase prioritizing flexibility, but there are optimization opportunities that preserve the experimental design.

**Benchmark Results (10K points, Fashion-MNIST, single-threaded):**
- ManifoldANN HNSW: 15.56s build, 4805 QPS, 96.9% recall
- hnswlib: 2.96s build, 8117 QPS, 100% recall
- FAISS: 2.88s build, 9599 QPS, 100% recall

### Key Differences vs hnswlib/FAISS

**1. Memory Layout (30-40% speedup potential)**
- **Current:** Nested `Vector{Vector{Vector{Int}}}` structure causes cache misses and GC pressure
- **hnswlib/FAISS:** Flat contiguous arrays with offset indexing (`neighbors[offsets[node_id]]`)
- **Trade-off:** Medium implementation complexity, LOW flexibility impact

**2. Priority Queue for Candidates (20-30% speedup potential)**
- **Current:** Full sort on every candidate insertion (`sort!(candidates, by = c -> c.dist)`)
- **hnswlib/FAISS:** MinMaxHeap with cached worst-distance for O(log k) operations
- **Trade-off:** Medium implementation complexity, NO flexibility impact

**3. Neighbor Selection Heuristic (15-25% speedup potential)**
- **Current:** Simple greedy "keep M closest" strategy
- **FAISS:** Connectivity-aware pruning (reject dominated neighbors)
- **Trade-off:** Low implementation complexity, NO flexibility impact
- **Note:** Could implement as alternate `AbstractNeighborPolicy` variant

**4. SIMD Distance Functions (10-20% speedup potential)**
- **Current:** Generic distance function with runtime polymorphism
- **hnswlib/FAISS:** Hand-coded SSE/AVX intrinsics
- **Trade-off:** HIGH flexibility impact - **NOT RECOMMENDED** (defeats thesis goal of metric flexibility)

**5. Parallel Construction (4-8× speedup potential)**
- **Current:** Sequential insertion
- **hnswlib:** Lock-striped parallel insertion (65K lock bins)
- **Trade-off:** Very high complexity, non-deterministic graphs - **NOT RECOMMENDED** for research

### Applied Optimizations (Already Implemented)

✅ **Use `partialsort!` instead of `sort!`** in `select_neighbors` and `_prune_list!`
- Expected: 10-15% speedup per location
- Actual: Minimal impact (bottleneck is elsewhere)

✅ **Remove unnecessary list assignments** in `_link_nodes!`
- Adjacency lists already modified in-place, no need for `adjacency[a] = list_a`
- Expected: 5-10% speedup
- Actual: Minimal impact

✅ **Multithreading for batch queries**
- Query performance: 4805 QPS single-threaded, competitive with alternatives
- Build time: Still sequential

### Recommended Future Work

**Priority 1 (Medium effort, high value):**
- [ ] Implement flat memory layout for adjacency lists
  - Replace `Vector{Vector{Vector{Int}}}` with flat storage + offsets
  - Expected: 30-40% build time reduction
  - Preserves all flexibility for testing neighbor policies

**Priority 2 (Medium effort, high value):**
- [ ] Implement MinMaxHeap for candidate management in `_search_layer`
  - Replace repeated sorting with heap-based priority queue
  - Expected: 20-30% build time reduction
  - No flexibility impact, just better data structure

**Priority 3 (Low effort, medium value):**
- [ ] Implement connectivity-aware neighbor selection as `DiversityNeighborPolicy`
  - Alternative to `HeuristicNeighborPolicy` for comparison
  - Expected: 15-25% build time reduction
  - Fits naturally into existing `AbstractNeighborPolicy` design

**Not Recommended:**
- ❌ SIMD-specialized distances (loses metric flexibility)
- ❌ Parallel construction (complexity not worth it for research use case)

### Combined Impact Estimate

Implementing Priority 1 + 2 would yield ~60-80% speedup:
- Current: 15.56s
- After optimization: ~7-9s
- Remaining gap to hnswlib/FAISS: ~2.5-3× (acceptable for research codebase)

The remaining gap is due to:
1. C++ inherent advantages (manual memory control, zero-cost abstractions)
2. Years of production micro-optimizations (prefetching, cache-line alignment)
3. Acceptable trade-off for maintainability and flexibility

### Additional Context: Recall Gap Investigation

Current recall: 96.9% vs 100% for hnswlib/FAISS (same parameters: M=16, ef_construction=200, ef_search=50)
- Indicates potential correctness bug in graph construction or search traversal
- Should be investigated before major performance refactoring
- See query.jl:235-241 (pruning logic) and neighbor_policy.jl:20-26 (selection strategy)
