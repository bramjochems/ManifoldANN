# Future work

Open items, working principles, and dead-ends to avoid revisiting.
Completed work lives in `git log` — this file is forward-looking only.

## Working principles

These are invariants we've established the hard way. Violating them
silently breaks something.

- **Thread-safety contract on `AbstractANNIndex`**: `query(::AbstractANNIndex, data, q, k)`
  is concurrent-safe in every concrete implementation today, but it's a happy
  coincidence rather than a documented invariant. New index types must preserve
  it. The contract: `query` may be called concurrently on the same index;
  `build_index` and `insert!` may not. Worth promoting to a docstring on
  `AbstractANNIndex` and adding a generic regression test.

- **Distance functions must be re-entrant.** The threaded build path and the
  batch query path call `index.distance` concurrently from multiple workers.
  Stateful distance functors (e.g. with internal cache) silently corrupt under
  these paths. Documented on `HNSWIndex`; should be hoisted to the abstract
  index type.

- **Do not use `LoopVectorization.jl` (`@turbo` / `@avx`).** Maintenance-only
  since ~2024, fragile on Julia 1.10+. Stick to `@simd` + `SIMD.jl` +
  `VectorizationBase.jl` if hand-tuning is needed.

- **Do not measure perf in the test suite.** `Pkg.test()` is the correctness
  gate; perf goes in `scripts/*_bench.jl`. Conflating them makes CI flaky and
  obscures regressions.

- **Report testset count, not raw `@test` count.** Per-edge assertions in
  loops inflate the latter into meaningless six-digit numbers; the meaningful
  signal is "X testsets passed, 0 failed."

## Open work

### HNSW

- **Fix stale `cur_max` snapshot in threaded build.** A thread that snapshots
  `cur_max=2` at insertion start can have its connect loop skip layers 3..level
  if another thread bumps `index.max_layer` past `level` mid-execution. Same
  effect as serial HNSW would have produced for that node, so locally
  HNSW-correct, but a small recall hit (~0.001 absolute, ~0.1% of inserts).
  Approach: hold `index.global_lock` for the connect loop *only* when
  `level > cur_max_at_snapshot`. A previous fix attempt collapsed recall to
  ~0.5 from a bug I couldn't quickly debug — re-attempt carefully with the
  stress + invariant tests as a guard rail.

- **Lock-free adjacency + visited-buffer pool (review ROI before committing).**
  Two pieces that together would close most of the remaining threading gap to
  hnswlib (estimated multi-thread build 3.96s → ~2.0s, multi-thread query
  16k → ~30-40k QPS at SIFT-128 n=50k). Slab adjacency: replace
  `Vector{Vector{Int}}` with `Matrix{Int}` + atomic degree counters, eliminate
  per-node `ReentrantLock`. Visited pool: lift `BatchQueryScratch`-style
  buffers from per-batch to index-lifetime ownership, atomic-stack pool.

  Cost ~2-3 days. Won't reach hnswlib parity (C++ vs Julia constant factors
  cap us at ~1.4× behind, not 1.0×). Re-evaluate after thesis whether library
  publishability needs the extra speed-up or whether "competitive but
  Julia-native" is sufficient.

- **Pool the `BestCandidatesHeap` backing buffer for build.** Per-call heap
  allocation in `_search_layer` is the dominant remaining single-threaded
  build alloc (~36 KB/call at ef_c=200). Same coupling tradeoff as
  `visit_stamps` (build-only, single-threaded). Estimated 10-20% single-thread
  build speedup. Smaller win than the threaded build work.

- **HNSW traversal `maybe_push_candidate!` deviates from the paper.** Currently
  pushes to `state.pending` only when also added to `best`. The standard HNSW
  search adds *every* unvisited neighbor whose `dist < worst` to the candidate
  frontier. Likely benign but worth aligning. `src/indices/hnsw/traversal.jl:117-127`.

### NN-Descent

- **Adaptive `max_iterations`.** Currently hardcoded at 10; PyNNDescent uses
  `max(5, round(log2(n)))`. Cheap fix, plausibly small recall improvement at
  large n.

- **PyNNDescent as a quality oracle.** Head-to-head validation against
  PyNNDescent on identical inputs would catch creeping recall drops the
  current ≥ 0.90 / MRR ≥ 0.92 floor doesn't notice. Costs Python in the
  validation loop; defer until the ecosystem question (below) is decided.

### LSH

- **`_collect_candidates` dedup is wrong.** `src/indices/lsh/index.jl:133-151`
  uses sort+unique (O(C log C) where BitSet is O(C)) and the post-dedup
  `resize!(candidates, candidate_cap)` keeps lowest-numbered IDs after sort —
  not a meaningful subset. Fix: BitSet for dedup; either don't truncate
  (compute distances for all then top-k) or sample uniformly.

- **`pack_bins(bins) = UInt64(hash(Tuple(bins)))`** allocates a Tuple per call
  on the inner LSH hash path. `src/indices/lsh/hash_functions.jl:52`. Fold a
  simple FNV/xxhash over the Int projections directly.

### Geodesic / graph

- **`pushfirst!(path, current)` in `_reconstruct_path`.**
  `src/geodesic/geodesic_model.jl:288`. O(n) per call → O(n²) reconstruction.
  Use `push!` then `reverse!` once.

- **`GreedySolver` complexity claim wrong.**
  `src/graphs/refinement/solvers.jl:459` documents O(k² log k); actual is
  O(k⁴) worst case. Either fix the algorithm (heap of edges) or fix the
  docstring.

- **`_fit_geometries(::ShareSimilarTangents)` boxing.** Both sharing variants
  use `Vector{Any}` because they fill out of order with `nothing` placeholders.
  Could tighten to `Vector{Union{Nothing, G}}` once `G` is known after the
  first fit. Low-priority polish.

### New indices to consider

- **`RPTreeIndex` as a standalone index.** RP-tree primitives already exist in
  the codebase (`src/indices/nndescent/rptree_init.jl`: `build_rptree`,
  `build_rptree_forest`, `leaf_members`) and are used as an opt-in init for
  NN-Descent. Promoting them to a top-level `RPTreeIndex` would be cheap (~50
  LOC of wrapping struct + `build_index` / `query` methods) and covers the
  high-d regime where our KD-tree degrades (>~50 dims) — RP-trees pick random
  hyperplanes and avoid the axis-aligned curse-of-dimensionality. Modest novel
  contribution in the Julia ecosystem: `NearestNeighbors.jl` ships
  KDTree/BallTree/BruteTree but not RP-tree forest; the only Julia RP-tree
  implementation today is buried inside `NearestNeighborDescent.jl`'s init.
  When this lands, move the primitives from `src/indices/nndescent/` to
  `src/utils/rptree.jl` so NN-Descent and `RPTreeIndex` share them.

### Tooling

- **Re-enable PyNNDescent in the benchmark runner.** Wrapper exists at
  `benchmarking/benchmarking/wrappers/pynndescent.py` but excluded from
  `benchmarking/configs/algorithms.yaml` with a stale comment about Python <3.10.
  Comment is wrong: a recent install on Python 3.13.5 venv (`uv pip install
  pynndescent`) succeeded with `llvmlite==0.47.0`, `numba==0.65.1`,
  `pynndescent==0.6.0`. Re-enable; drop the stale comment.

- **Benchmark against NearestNeighbors.jl (KDTree at minimum, ideally
  BallTree).** A wrapper for `NearestNeighbors.KDTree` already exists at
  `benchmarking/benchmarking/wrappers/julia_external.py` and is registered, but
  is not enabled in any config in `benchmarking/configs/*.yaml`. No BallTree
  wrapper at all. Pairings worth adding:
  - `MANN-KDTree` vs `NearestNeighbors-KDTree`: direct head-to-head on tree
    construction + query.
  - `MANN-BruteForce` vs `NearestNeighbors-BallTree`: BallTree should beat
    KD-tree at d > 50; useful sanity check on whether we should add BallTree
    as an index too.
  - `MANN-NNDescent` vs `NearestNeighborDescent-jl` (already configured)
    remains the meaningful NN-Descent pairing.
  Won't cover all our indices (no BallTree on our side, no LSH or HNSW in
  NearestNeighbors.jl), but a few well-chosen pairs would tell us where we
  stand against the dominant Julia ANN library.

### Areas not reviewed

The independent code review skipped: `src/indices/multilevel/*` (IVF-HNSW),
`src/transforms/kmeans/*`, `src/preprocessing/*`,
`src/geodesic/refinement.jl`, `src/geometry/neighborhood.jl`,
`src/geometry/criteria.jl`, ORC `EdgeNeighborhoodView` construction
(`graphs/refinement/{types,filtering,effective_epsilon_policy}.jl`). Worth a
second pass before claiming the package has been comprehensively reviewed.

## Strategic decisions outstanding

### Julia-ecosystem-native vs self-contained

Three workstreams point in the same direction and should be decided
together, not as discrete fixes:

- **Distances.jl as the metric provider.** Replace `default_distance` etc.
  with thin aliases over `Euclidean()` / `SqEuclidean()` / etc. Wire
  `Distances.pairwise!` into bruteforce builder/query and k-means Lloyd
  kernels — that's where the BLAS3 win materialises (cross-term identity
  dispatches to GEMM). Per-call `evaluate(metric, x, y)` is **no** SIMD win
  over our current `@simd` kernels — only bulk pairwise/colwise pays. Half a
  day for the alias swap; 2-3 days for `pairwise!` integration with
  benchmarks. Risk: BLAS-backed pairwise can produce small negatives from
  cancellation; bit-for-bit results may differ from `@simd` reduction and
  could perturb tie-breaking in unit tests.

- **Manifolds.jl as the manifold/sampler provider.** Wire as a sampler +
  ground-truth-distance backend. Adapter: `sample_manifold(M, n) → (ambient,
  intrinsic)`, ~30 LOC per manifold, built on `rand(M)` / `embed(M, p) |>
  vec` / `distance(M, p, q)`. Concrete manifolds worth adding for the
  experimental story: $S^2$, $H^2$, $\mathrm{Gr}(4,2)$, SPD $P(3)$
  (vech-flattened), $SO(3)$. Friction: typed-point representations vary
  (SPD is `Symmetric{Matrix}`, Grassmann is `Matrix`, etc.) — the
  ambientize layer must flatten via `embed(M,p) |> vec`. EmbeddedTorus has
  no closed-form geodesic in Manifolds.jl either; the existing torus would
  still need grid-Dijkstra. Local PCA tangent estimation degrades when
  ambient dim ≫ intrinsic dim or when the true tangent space is an affine
  subspace of a non-linear variety (SPD): the Manifolds.jl-exact-tangent
  vs local-PCA comparison would actually strengthen the empirical
  narrative. Dependency cost: Manifolds.jl + ManifoldsBase.jl +
  RecursiveArrayTools + StaticArrays + Distributions + ManifoldDiff
  (~3-5s load time). Effort: 1-2 weeks.

- **NearestNeighbors.jl as an alternative ANN backend.** Mature, threaded,
  gives us BallTree (useful for higher-d regimes where KD-tree degrades).
  Replacing our KD-tree for parity is not worth it; adding a
  `NearestNeighborsBallTree` index wrapper would be cheap and additive.

The thesis benefits from the current self-contained stance (clear scope,
fewer moving parts). A published library benefits from ecosystem-native
(less code to maintain, broader compatibility, easier for others to extend).
**Decide direction first**, then sequence the three as one coherent effort.

## Anti-patterns (don't re-explore these)

One-liners pointing at full context in commit history.

- **NN-Descent dict-based dedup variant**: 18-35% slower than baseline at
  k∈{15,30,50}. Set/hash overhead exceeds the cheap O(k) scan at typical k.
  See `d9bab8a` for the simpler simplification that did land.

- **NN-Descent `_update_neighbor_dist!` "asymptotic fix"**: the catastrophic
  branch is unreachable on deterministic distances (same `(a,b)` always
  yields the same distance, so `dist < existing.dist` never fires). The
  perf TODO that flagged it was based on an unreachable path. See `d9bab8a`.

- **RP-tree initialization for NN-Descent**: did NOT close the build gap.
  All three implementation variants tested were 1.6-10× *slower* than random
  init. Reason: with bidirectional random init, NN-Descent converges in 1-2
  iterations on this codebase's `convergence_threshold=0.001` default — no
  iterations to save. Did improve recall (n=5000 d=32 k=15: 0.92 → 0.96,
  +3.6pp). Retained as opt-in via `init=:rptree`. See `05b91d3`.

- **Threading rework via `@sync` + `@spawn` over chunked ranges**: tried as
  a fix for "30% idle" profile result, ended 6-9% slower. The "idle" was
  task-switching, not thread starvation. Per-task alloc + spawn/sync cost
  exceeded the load-imbalance savings at our work granularity.

- **Edge-weight threading deferred deliberately**: edge weights are ~10% of
  graph build time (PCA fit dominates), so Amdahl caps the win at ~10%, AND
  `compute_edge_weight` is a public extension point — silently threading
  the outer loop would impose an undocumented thread-safety contract on
  third-party `AbstractEdgeWeight` / `AbstractLocalGeometry` implementations.
  Revisit only after formalising the thread-safety contract for those
  extension points. If pursued, expose as opt-in.

- **`ShareSimilarTangents` threading**: semantically serial. Each iteration's
  share/fit decision depends on which earlier nodes were fitted vs shared.
  Threading would lose the sharing structure or produce schedule-dependent
  outputs. Leave serial by design.
