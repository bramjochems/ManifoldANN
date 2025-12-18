# TODO: Integration with Existing Julia Packages

This document tracks integration opportunities with established Julia packages to improve code quality, performance, and maintainability.

## Priority: HIGH

### 1. Graphs.jl Integration

**Current State:**
- We have custom graph structures in `src/graphs/weighted_knn_graph.jl`
- Custom shortest path implementations in `src/geodesic/`
- Custom graph utilities

**Goal:**
- Use Graphs.jl as the underlying graph representation
- Leverage optimized graph algorithms from the ecosystem
- Maintain compatibility with our weighted edge semantics

**Tasks:**

#### 1.1 Replace Custom Graph with Graphs.jl Backend
- [ ] Study Graphs.jl API (SimpleGraph, SimpleDiGraph, MetaGraphs.jl)
- [ ] Decide: Use MetaGraphs.jl for edge weights or parallel weight array?
- [ ] Create adapter/wrapper type: `WeightedKNNGraph <: AbstractGraph`
- [ ] Implement Graphs.jl interface methods:
  - [ ] `nv(g)` - number of vertices
  - [ ] `ne(g)` - number of edges
  - [ ] `vertices(g)` - vertex iterator
  - [ ] `edges(g)` - edge iterator
  - [ ] `has_edge(g, s, d)` - edge existence check
  - [ ] `outneighbors(g, v)` - neighbor list
  - [ ] `inneighbors(g, v)` - for directed graphs
- [ ] Add edge weight accessor compatible with Graphs.jl patterns
- [ ] Write conversion utilities: `to_graphs_jl(wg)`, `from_graphs_jl(g, weights)`

**Files to modify:**
- `src/graphs/weighted_knn_graph.jl` - main changes
- `src/geodesic/distance_model.jl` - update to use Graphs.jl API
- Add `src/graphs/graphs_integration.jl` - integration layer

**Benefits:**
- Access to optimized graph algorithms (BFS, DFS, connected components)
- Better visualization via GraphPlot.jl
- Interoperability with graph ML ecosystem
- Community-maintained performance improvements

**References:**
- Graphs.jl: https://github.com/JuliaGraphs/Graphs.jl
- MetaGraphs.jl: https://github.com/JuliaGraphs/MetaGraphs.jl
- GraphsMatching.jl, GraphsFlows.jl for advanced algorithms

#### 1.2 Use Graphs.jl Shortest Path Algorithms
- [ ] Benchmark Graphs.jl dijkstra vs our implementation
- [ ] Test: `dijkstra_shortest_paths(g, src, weights)`
- [ ] Adapt our `geodesic_distance` to use Graphs.jl backend
- [ ] Compare performance on various graph sizes (n=1k, 10k, 100k)
- [ ] Decide: keep custom implementation or switch entirely?

**Hypothesis:** Graphs.jl will be faster for large graphs due to better optimizations.

#### 1.3 Graph Quality Metrics via Graphs.jl
- [ ] Use `connected_components(g)` to check connectivity
- [ ] Compute graph diameter: `diameter(g)`
- [ ] Average path length: `mean(shortest_paths(g))`
- [ ] Clustering coefficient: use from GraphMeasures.jl
- [ ] Degree distribution analysis

**New file:** `src/graphs/quality_metrics.jl`

---

## Priority: MEDIUM

### 2. Neighbourhood.jl Performance Comparison

**Current State:**
- We have custom k-NN implementations (BruteForce, KDTree, HNSW, etc.)
- No systematic benchmarking against other Julia packages

**Goal:**
- Understand performance vs Neighbourhood.jl
- Identify opportunities for improvement
- Validate our implementation choices

**Tasks:**

#### 2.1 Benchmark Setup
- [ ] Install Neighbourhood.jl: `using Pkg; Pkg.add("Neighbourhood")`
- [ ] Create benchmark suite in `benchmarks/vs_neighbourhood/`
- [ ] Test datasets:
  - [ ] MNIST (784D, 60k points)
  - [ ] SIFT (128D, 1M points)
  - [ ] Swiss roll (3D, 10k points)
  - [ ] Random (100D, varying n)
- [ ] Metrics to compare:
  - [ ] Index build time
  - [ ] Query time (single query, batch queries)
  - [ ] Memory usage
  - [ ] Recall @ k=10 vs brute force

**New file:** `benchmarks/vs_neighbourhood/benchmark_suite.jl`

#### 2.2 BruteForce Comparison
- [ ] Compare our `BruteForceIndex` vs `BruteTree` from Neighbourhood.jl
- [ ] Check: do they use same distance computation optimizations?
- [ ] Profile both implementations
- [ ] Identify bottlenecks in our code

**Expected outcome:** Should be similar; if not, learn from their optimizations.

#### 2.3 KDTree Comparison
- [ ] Compare our KDTree vs NearestNeighbors.jl KDTree
- [ ] Test on low-D (d<10) and medium-D (d=50) data
- [ ] Compare tree construction strategies
- [ ] Check splitting heuristics (midpoint vs median)

#### 2.4 Learn from Neighbourhood.jl
- [ ] Study their distance function interface
- [ ] Check for SIMD optimizations we're missing
- [ ] Review their batch query implementation
- [ ] Look for numerical stability improvements

**Deliverable:**
- Performance report in `benchmarks/vs_neighbourhood/RESULTS.md`
- List of actionable improvements for our code

---

## Priority: MEDIUM

### 3. PCA from Existing Packages

**Current State:**
- Custom PCA implementation in `src/preprocessing/pca_transform.jl`
- Used for local geometry estimation and dimensionality reduction

**Goal:**
- Use battle-tested PCA from MultivariateStats.jl or similar
- Reduce maintenance burden
- Improve numerical stability

**Tasks:**

#### 3.1 Evaluate PCA Packages

**Options:**
1. **MultivariateStats.jl** - Standard choice, well-maintained
   - `fit(PCA, data; maxoutdim=k)`
   - SVD-based, numerically stable
   - Supports incremental PCA

2. **StatsAPI.jl** - Common interface, flexible
   - Multiple backend options
   - Good for abstraction

3. **Manifolds.jl** - Geometric perspective
   - Tangent space projections
   - Riemannian optimization

**Tasks:**
- [ ] Install and test MultivariateStats.jl
- [ ] Compare API with our custom PCA
- [ ] Check if it supports:
  - [ ] Centering (subtract mean)
  - [ ] Explained variance ratios
  - [ ] Projection and reconstruction
  - [ ] Incremental updates (for streaming data)

#### 3.2 Replace Custom PCA Implementation
- [ ] Create wrapper: `ManifoldPCA` that uses MultivariateStats.jl
- [ ] Update `src/preprocessing/pca_transform.jl`:
  ```julia
  using MultivariateStats: PCA, fit, transform, reconstruct

  struct ManifoldPCA <: PreprocessingTransform
      model::PCA
      # ... existing fields
  end

  function fit_pca(data::AbstractMatrix; intrinsic_dim::Int)
      # Use MultivariateStats.jl
      pca_model = fit(PCA, data; maxoutdim=intrinsic_dim)
      return ManifoldPCA(pca_model, ...)
  end
  ```
- [ ] Update tests to ensure same behavior
- [ ] Benchmark: custom vs MultivariateStats.jl
- [ ] Keep custom implementation as fallback if needed

**Files to modify:**
- `src/preprocessing/pca_transform.jl`
- `src/geodesic/local_geometry.jl` (uses PCA for tangent planes)
- `test/unit/preprocessing/pca_transform_tests.jl`

#### 3.3 Local PCA for Dimensionality Estimation
- [ ] Review how we use PCA in `LocalGeometryEstimator`
- [ ] Consider using MultivariateStats.jl for:
  - [ ] Computing eigenvalues of local neighborhoods
  - [ ] Explained variance for intrinsic dimension estimation
  - [ ] Tangent space fitting
- [ ] Ensure we can still access:
  - [ ] Individual eigenvalues (for gap analysis)
  - [ ] Eigenvectors (for tangent plane basis)
  - [ ] Reconstruction error (for adaptive neighborhoods)

**Benefit:** More robust numerical behavior for ill-conditioned covariance matrices.

---

## Priority: LOW (Nice to Have)

### 4. Additional Integration Opportunities

#### 4.1 NearestNeighbors.jl
- [ ] Compare their KDTree, BallTree implementations
- [ ] Check if we should use their spatial indexing
- [ ] Evaluate Metric interface compatibility

#### 4.2 Distances.jl
- [ ] Use standard distance metrics from Distances.jl
- [ ] Replace our custom `default_distance` with `Euclidean()`
- [ ] Support cosine, Minkowski, etc. via Distances.jl API
- [ ] Ensure type stability with parametric distance types

#### 4.3 LinearMaps.jl for Random Projections
- [ ] Represent random projection matrices as LinearMaps
- [ ] Avoid storing dense projection matrices
- [ ] Use structured random projections (FJLT, sparse JL)

#### 4.4 Manifolds.jl Integration
- [ ] Explore Manifolds.jl for geodesic computations
- [ ] Use for non-Euclidean manifolds (sphere, hyperbolic space)
- [ ] Potential for future generalization beyond Euclidean embedding

#### 4.5 GraphPlot.jl for Visualization
- [ ] Once we use Graphs.jl, add visualization utilities
- [ ] Plot k-NN graphs with edge weights
- [ ] Visualize graph quality (connected components, etc.)

**New file:** `src/visualization/graph_plots.jl` (optional)

---

## Testing Strategy

For each integration:

1. **Compatibility Tests**
   - [ ] Ensure existing functionality unchanged
   - [ ] Test edge cases (empty graphs, single point, etc.)
   - [ ] Validate numerical accuracy

2. **Performance Tests**
   - [ ] Benchmark before and after
   - [ ] Profile memory allocations
   - [ ] Check for regressions

3. **Documentation**
   - [ ] Update docstrings to mention underlying package
   - [ ] Add examples showing how to use new features
   - [ ] Document any breaking changes

---

## Migration Plan

### Phase 1: Research (1 week)
- [ ] Install and explore all candidate packages
- [ ] Read documentation and examples
- [ ] Run small experiments to understand APIs

### Phase 2: Graphs.jl Integration (2 weeks)
- [ ] Implement Graphs.jl adapter
- [ ] Update geodesic distance code
- [ ] Add graph quality metrics
- [ ] Test thoroughly

### Phase 3: PCA Integration (1 week)
- [ ] Replace custom PCA with MultivariateStats.jl
- [ ] Update local geometry code
- [ ] Verify numerical equivalence

### Phase 4: Benchmarking (1 week)
- [ ] Create comprehensive benchmark suite
- [ ] Compare vs Neighbourhood.jl
- [ ] Document findings
- [ ] Identify optimization opportunities

### Phase 5: Cleanup (ongoing)
- [ ] Remove redundant code
- [ ] Update documentation
- [ ] Add integration tests
- [ ] Consider deprecation warnings for old APIs

---

## Open Questions

1. **Graphs.jl edge weights:** MetaGraphs.jl vs parallel array?
   - MetaGraphs: More idiomatic, but adds dependency
   - Parallel array: Lighter weight, current approach
   - **Decision needed:** Profile both approaches

2. **Distance metrics:** Abstract via Distances.jl or keep simple?
   - Pro: Flexibility, standard interface
   - Con: Type stability concerns, complexity
   - **Decision needed:** Start simple, generalize later?

3. **PCA incremental updates:** Do we need streaming PCA?
   - Current use case: offline, batch processing
   - Future use case: online manifold learning?
   - **Decision needed:** Defer until needed

4. **Breaking changes:** How to handle API changes?
   - Option 1: Major version bump (JuliANN v2.0)
   - Option 2: Gradual deprecation warnings
   - **Decision needed:** Check with project goals

---

## Success Criteria

Integration is successful when:

- [ ] All tests pass with new dependencies
- [ ] Performance is equal or better than before
- [ ] Code is simpler and more maintainable
- [ ] Documentation is updated
- [ ] No regressions in functionality
- [ ] Package dependencies are well-justified

---

## References

- **Graphs.jl:** https://juliagraphs.org/Graphs.jl/stable/
- **MultivariateStats.jl:** https://github.com/JuliaStats/MultivariateStats.jl
- **Neighbourhood.jl:** https://github.com/JuliaNeighbours/Neighbourhood.jl
- **NearestNeighbors.jl:** https://github.com/KristofferC/NearestNeighbors.jl
- **Distances.jl:** https://github.com/JuliaStats/Distances.jl
- **Manifolds.jl:** https://juliamanifolds.github.io/Manifolds.jl/stable/

---

**Last Updated:** 2025-12-13
**Status:** Planning phase
**Owner:** Integration team
