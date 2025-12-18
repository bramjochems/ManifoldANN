# TODO

This document tracks improvement tasks for ManifoldANN, organized by priority. Items are categorized as High (do soon), Medium (nice to have), or Low (polish/future work).

## Complete NN-Descent Symmetrization Investigation

**Context:** On at least one dataset, **deferred symmetrization outperforms continuous symmetrization** in terms of recall. This is unexpected and needs investigation.

**Possible Explanations:**
1. **Premature Pruning**: Continuous symmetry with `PrunedSymmetry(1.5)` prunes to 1.5×k after each iteration, potentially discarding valuable neighbors
2. **New/Old Neighbor Tracking**: Continuous symmetrization constantly adds reverse edges, which might confuse new/old tracking
3. **Heap Capacity Pressure**: Continuous addition of reverse edges may cause good candidates being rejected due to fuller heaps
4. **Convergence Threshold Interaction**: Algorithm stops when `updates / (n * k) < 0.01`; continuous symmetry might cause early termination


## Extract Long Functions

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
