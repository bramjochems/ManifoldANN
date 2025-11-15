# TODO

This document tracks improvement tasks for ManifoldANN, organized by priority. Items are categorized as High (do soon), Medium (nice to have), or Low (polish/future work).

---

## 🔴 High Priority

### Complete NN-Descent Symmetrization Investigation

**Context:** On at least one dataset, **deferred symmetrization outperforms continuous symmetrization** in terms of recall. This is unexpected and needs resolution before publication.

**Possible Explanations:**
1. **Premature Pruning**: Continuous symmetry with `PrunedSymmetry(1.5)` prunes to 1.5×k after each iteration, potentially discarding valuable neighbors
2. **New/Old Neighbor Tracking**: Continuous symmetrization constantly adds reverse edges, which might confuse new/old tracking
3. **Heap Capacity Pressure**: Continuous addition of reverse edges may cause good candidates being rejected due to fuller heaps
4. **Convergence Threshold Interaction**: Algorithm stops when `updates / (n * k) < 0.01`; continuous symmetry might cause early termination

**Investigation Tasks:**
- [ ] Track iteration counts: Compare when continuous vs deferred variants stop iterating
- [ ] Log update counts per iteration: Check if continuous symmetry reduces updates per iteration
- [ ] Test on all datasets: Determine if this is dataset-specific or universal
- [ ] Compare full-continuous vs full-deferred: Check if effect is specific to pruned symmetry
- [ ] Try intermediate approaches: Apply symmetry every 2-3 iterations instead of every iteration
- [ ] Analyze final graph properties: Compare degree distributions, average path lengths
- [ ] Profile heap rejection rates: Count how often `push!` fails due to capacity

**Questions to Answer:**
1. Is this effect consistent across all datasets or specific to certain data distributions?
2. Does full symmetry show the same behavior or is it specific to pruned symmetry?
3. What is the optimal frequency for applying symmetry (every iteration vs every N iterations)?
4. Can we predict which symmetry timing will work better for a given dataset?

**Related Files:**
- `src/indices/nndescent/builder.jl` - Core NN-Descent implementation
- `benchmarking/configs/*.yaml` - Dataset configurations with variants

---

## Performance Characteristics

| Index | Build Time | Query Time | Memory | Recall | Supports Mutation |
|-------|-----------|------------|---------|--------|------------------|
| BruteForce | O(1) | O(nd) | O(1) | 100% | Yes |
| KDTree | O(n log n) | O(log n) avg | O(n) | 100% | No |
| LSH | O(nL) | O(nL/B) | O(nL) | 60-90% | Yes |
| HNSW | O(n log n) | O(log n) | O(nM) | 90-99% | Yes |
| IVF+HNSW | O(n log n) | O(k + log(n/k)) | O(n) | 85-95% | Via children |

Where: L = #tables, B = avg bucket size, M = max neighbors, k = #clusters

## Thread Safety

**Thread-safe operations:**
- All `query` operations (read-only)
- Batch queries (automatically parallelized)
- Concurrent reads from same index

**NOT thread-safe:**
- `insert!` operations (requires external locking)
- Concurrent `build_index` + `query`
- Modifying index parameters after construction

---

## 🟡 Medium Priority

### Extract Long Functions

**Functions exceeding 100 lines that need decomposition:**

**src/indices/nndescent/builder.jl:**
- [ ] `_run_nndescent!` (~136 lines)
  - Extract `_evaluate_new_new_pairs!`
  - Extract `_evaluate_new_old_pairs!`
  - Extract `_apply_iteration_symmetry!`

**src/indices/hnsw/query.jl:**
- [ ] Add complexity comments to layer-aware search
  - Document algorithmic phases
  - Explain layer traversal strategy

**src/indices/multilevel/query.jl:**
- [ ] `_query_recursive` (~150 lines)
  - Extract routing logic
  - Extract result aggregation
  - Add phase documentation

**Benefits:**
- Improved maintainability
- Easier unit testing
- Better code readability
- Clearer algorithmic structure

---

### HNSW Flat Memory Layout Optimization

**Context:** Current HNSW build time is ~5× slower than hnswlib/FAISS (4s vs 0.7s for 10K points). Nested vector structure causes cache misses and GC pressure.

**Current:** `Vector{Vector{Vector{Int}}}` structure
**Proposed:** Flat contiguous arrays with offset indexing

**Expected Benefits:**
- 30-40% build time reduction
- Better cache locality
- Reduced GC pressure
- Preserves all flexibility for testing neighbor policies

**Trade-offs:**
- Medium implementation complexity
- Low flexibility impact
- Maintains experimental design goals

**Tasks:**
- [ ] Design flat storage format with offset arrays
- [ ] Implement neighbor access via offsets
- [ ] Benchmark before/after performance
- [ ] Verify recall unchanged
- [ ] Update documentation

**Not Recommended:**
- ❌ SIMD-specialized distances (loses metric flexibility)
- ❌ Parallel construction (complexity not worth it for research)

---

### NN-Descent Performance Improvements

**Context:** PyNNDescent outperforms current implementation due to higher-quality initialization, diversification, and parallel machinery.

**Potential Improvements:**
- [ ] Tune `max_candidate_neighbors` defaults per dataset/metric
  - Document recommended settings
  - Add auto-tuning heuristics

- [ ] Retain neighbor distances through construction
  - Enable diversification/pruning pass
  - Add angular check similar to PyNNDescent

- [ ] Add optional RP-tree or projection-tree seeding
  - Improve initialization quality
  - Shrink required iterations

- [ ] Experiment with incremental symmetry
  - Apply symmetry every N iterations
  - Balance between continuous and deferred

- [ ] Explore parallel candidate evaluation
  - Use `Threads.@threads` for pair evaluation
  - Requires thread-safe data structures
  - Modulo-based partitioning scheme

---


## 🟢 Low Priority (Polish / Future Work)
### Multi-Level Index Optimizations (Deferred)

**Deferred Until Profiling Shows Bottleneck:**

**Batch Transform API:**
- [ ] Add `transform_batch(transform::AbstractTransform, X::Matrix)` interface
  - Enable vectorized implementations
  - Example: KMeans compute `pairwise_distances!(X, centroids)` once
  - Requires handling heterogeneous `TransformResult` types

- [ ] Implement `transform_batch` for KMeansTransform
  - Compute all point-to-centroid distances in single pass
  - Eliminates n separate `compute_distances` calls
  - Expected speedup: 10-30% for expensive distance computations

- [ ] Extend `apply_transform_batch` to use optional batch method
  - Fall back to per-column loop if batch method not available
  - Maintains backward compatibility

**When to Revisit:**
- If profiling shows `partition_by_transform` is >20% of build time
- When implementing PQ/OPQ transforms that benefit from batch processing
- If datasets grow large enough that allocation overhead dominates

---
