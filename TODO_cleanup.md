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

- **`benchmarking/` vs `scripts/` are different things.** `benchmarking/` is
  thesis-coupled experimentation infrastructure (downloaders, ann-benchmarks
  competitor wrappers, multi-algorithm sweep configs). It exists to produce
  thesis numbers/plots and will eventually migrate out of the package per
  `CLAUDE.md`'s separation principle. `scripts/` is package-coupled tooling
  (focused fair-compare benchmarks, perf regression checks, profiling) and
  stays with the package — anything a future consumer of `ManifoldANN` would
  find useful for understanding the package's behaviour. New work goes to
  whichever it serves; don't conflate them.

- **Thesis-grade head-to-head numbers come from focused scripts, not the
  general harness.** `benchmarking/benchmark.py` is for breadth ("roughly
  where do we sit across N algorithms × M datasets?"). Specific QPS / build
  time claims that get cited in the thesis or in commit messages must come
  from a focused fair-compare script in `scripts/` (e.g.
  `scripts/hnsw_fair_compare.py`, `scripts/kdtree_fair_compare.jl`). The
  general harness has known fairness issues — see "benchmark harness
  pre-migration fixes" under Open work.

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

### Benchmark harness pre-migration fixes (`benchmarking/`)

Independent fairness audit (mid-session, this conversation) flagged the
existing `benchmarking/benchmark.py` numbers as **not defensible** for
quantitative thesis-grade head-to-head comparisons against C/C++ libraries.
Recall numbers ARE defensible (consistent ground truth, consistent k, no
off-by-one). Build-time and QPS comparisons against external libraries
need fixing or re-running through focused scripts.

Three critical issues — fix these before relying on any harness build/qps
number. ManifoldANN-internal comparisons are mostly fair to each other
*except* for the HNSW JIT gap.

- **JIT not warmed for every Julia index type.**
  `benchmarking/benchmarking/wrappers/manifoldann.py:39-72` warms
  `BruteForceIndex` and `IVF-HNSW` only. `HNSWIndex`, `KDTreeIndex`,
  `LSHIndex` (both hash families), `IVFFlatIndex`, and `NNDescentIndex` pay
  full Julia JIT inside the timed `algo.fit()` region on first call.
  Same issue in `julia_external.py:50-53` for `HNSW.jl` and
  `NearestNeighborDescent.jl` (only `NearestNeighbors.KDTree` is warmed).
  Fix shape: extend each wrapper's warmup block to call `build_index` +
  `query_batch` for every index type the harness will exercise. Also the
  warmup uses dim=10; rerun warmup with the actual dataset dim before
  the first timed `fit` to cover dim-specialised code paths.

- **Competitor thread counts uncontrolled.**
  `wrappers/{hnswlib,faiss,annoy,pynndescent}.py` — no wrapper calls
  `set_num_threads`, `omp_set_num_threads`, or sets `n_jobs` /
  `NUMBA_NUM_THREADS`. hnswlib silently uses `min(cpus, 8)`, FAISS uses
  all OMP cores, pynndescent uses all numba threads. Meanwhile
  `benchmark.py:15-21` sets `JULIA_NUM_THREADS=cpu_count()` for the Julia
  side. **Result: cross-library comparisons are systematically asymmetric
  on threading.** Fix shape: a `--threads N` CLI flag that propagates to
  every wrapper's library-specific thread setter, plus require
  `JULIA_NUM_THREADS=N` to match.

- **Data marshalling charged asymmetrically.** `manifoldann.py:101-112`,
  `julia_external.py:66-74,185-193,274-282`. Every Julia wrapper does
  `np.asfortranarray(X.T, dtype=np.float32) + _to_matrix(...)` inside
  `fit()`. `wrappers/hnswlib.py:28-49` and `wrappers/faiss.py:26-57`
  consume row-major numpy directly with no equivalent step. For high-dim
  datasets this is tens to hundreds of ms charged to Julia only. Fix
  shape: marshal Julia inputs once outside the timed region (as
  `scripts/hnsw_fair_compare.py:189-196` already does), or symmetrically
  charge a `np.ascontiguousarray(X, dtype=np.float32)` to every
  competitor.

Lower-priority harness fixes (group when above is done):

- Query result conversion (Julia → 0-indexed Python ids) charged to Julia
  only — inside the `query_batch` timed region. `manifoldann.py:78-86,134-156`.
- pynndescent / numba JIT not warmed; first NNDescent call lands in
  timed build. Same for HNSW.jl `add_to_graph!` (the wrapper explicitly
  skips its warmup at `julia_external.py:57`).
- Single-shot timing, no variance, no `gc.collect()` / `GC.gc()` between
  algorithms — first algorithm pays cold-cache cost. `benchmark.py:344-367`.
  Fix: 1 untimed warm rep + 3 timed reps, report median, force GC between.
- No memory / allocation reporting — only wall time. Thesis claims about
  index footprint not backed by harness output.
- KDTree query is loop-of-singletons (`manifoldann.py:324-326` overrides
  `query_batch` to a Python loop), paying N×juliacall-boundary overhead
  vs SciPy's vectorised `cKDTree.query`. Algorithm-not-harness issue;
  flag rather than fix.

Two ann-benchmarks-style additions worth folding in once the harness is
fair:

- **Re-enable PyNNDescent in the benchmark runner.** Wrapper exists at
  `benchmarking/benchmarking/wrappers/pynndescent.py` but excluded from
  `benchmarking/configs/algorithms.yaml` with a stale comment about
  Python <3.10. Comment is wrong: recent install on Python 3.13.5 venv
  (`uv pip install pynndescent`) succeeded with `llvmlite==0.47.0`,
  `numba==0.65.1`, `pynndescent==0.6.0`. Re-enable; drop the stale comment.

- **Enable NearestNeighbors.jl in configs.** A wrapper for
  `NearestNeighbors.KDTree` exists at
  `benchmarking/benchmarking/wrappers/julia_external.py` and is registered,
  but is not enabled in any config in `benchmarking/configs/*.yaml`.
  No BallTree wrapper at all. The thesis got a focused one-off comparison
  via `scripts/kdtree_fair_compare.jl` (NN.jl-KDTree is ~2-4× faster on
  build, ~1.2-5× faster on query — informational, KDTree is reference-only
  for this thesis), but a config-level wiring would make it part of the
  routine harness output.

### Repository hygiene

- **`scripts/` directory cleanup.** Currently 43 files, mixed bag from
  multiple thesis workstreams (composite-shortcut research, ORC/curvature,
  geodesic, plus this session's perf scripts). Worth restructuring into
  subdirectories by purpose so it's coherent for a future consumer of the
  package. Suggested structure (defer concrete moves until the user can
  triage which research scripts are still active):
  - `scripts/perf/` — `hnsw_*`, `nndescent_*`, `kdtree_*`, `distance_micro.jl`
    (this session's diagnostic + fair-compare suite)
  - `scripts/research/` — composite-shortcut, ORC, geodesic experimental
    scripts (move to thesis-code repo if/when `benchmarking/` migrates)
  - `scripts/diagnostics/` — `diag_eff_eps.jl`, `diagnose_sinkhorn.jl`
  - `scripts/archive/` — already exists; sweep stale items in here
  - Top-level: `README.md`, `run_*.sh` orchestration scripts only
  - Also: `scripts/__pycache__/` is currently tracked; should be in
    `.gitignore`.

- **`benchmarking/` migration to a sibling thesis-code repo** (per
  `CLAUDE.md`'s separation principle: the package is the publishable
  library, thesis-coupled experiment infrastructure lives separately).
  This is a strategic move not blocking immediate work. Pre-requisite:
  the harness fairness fixes above, so the migrated harness isn't
  carrying known-broken numbers. After migration, the package's own
  perf surface lives in `scripts/perf/` (focused fair-compare scripts);
  the thesis-coupled multi-algorithm sweep harness lives in the sibling
  repo.

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
