# Future work

Open items, working principles, and dead-ends to avoid revisiting.
Completed work lives in `git log` — this file is forward-looking only.

## Working principles

Invariants we've established the hard way. Violating them silently breaks
something.

- **Thread-safety contract on `AbstractANNIndex`**: `query(::AbstractANNIndex, data, q, k)`
  is concurrent-safe in every concrete implementation today, but it's a
  happy coincidence rather than a documented invariant. New index types
  must preserve it. The contract: `query` may be called concurrently on
  the same index; `build_index` and `insert!` may not. Worth promoting to
  a docstring on `AbstractANNIndex` and adding a generic regression test.

- **Distance functions must be re-entrant.** The threaded build path and
  the batch query path call `index.distance` concurrently from multiple
  workers. Stateful distance functors (e.g. with internal cache) silently
  corrupt under these paths. Documented on `HNSWIndex`; should be hoisted
  to the abstract index type.

- **Do not use `LoopVectorization.jl` (`@turbo` / `@avx`).** Maintenance-only
  since ~2024, fragile on Julia 1.10+. Stick to `@simd` + `SIMD.jl` +
  `VectorizationBase.jl` if hand-tuning is needed.

- **Do not measure perf in the test suite.** `Pkg.test()` is the
  correctness gate; perf goes in `scripts/*_bench.jl`. Conflating them
  makes CI flaky and obscures regressions.

- **Report testset count, not raw `@test` count.** Per-edge assertions in
  loops inflate the latter into meaningless six-digit numbers; the
  meaningful signal is "X testsets passed, 0 failed."

- **`benchmarking/` vs `scripts/` are different things.** `benchmarking/`
  is thesis-coupled experimentation infrastructure (downloaders,
  ann-benchmarks competitor wrappers, multi-algorithm sweep configs). It
  exists to produce thesis numbers/plots and will eventually migrate out
  of the package per `CLAUDE.md`'s separation principle. `scripts/` is
  package-coupled tooling (focused fair-compare benchmarks, perf
  regression checks, profiling) and stays with the package — anything a
  future consumer of `ManifoldANN` would find useful for understanding
  the package's behaviour. New work goes to whichever it serves; don't
  conflate them.

- **Thesis-grade head-to-head numbers come from focused scripts, not the
  general harness.** `benchmarking/benchmark.py` is for breadth ("roughly
  where do we sit across N algorithms × M datasets?"). Specific QPS /
  build time claims that get cited in the thesis or in commit messages
  must come from a focused fair-compare script in `scripts/` (e.g.
  `scripts/hnsw_fair_compare.py`, `scripts/nndescent_jl_pareto.jl`,
  `scripts/kdtree_fair_compare.jl`).

- **Apples-to-apples library comparisons go on the recall-vs-qps Pareto
  curve, not at fixed parameter values.** Different libraries' search
  parameters (e.g. `ef_search` vs `max_candidates`) bound different data
  structures and dial recall at different rates. Numerically-equal
  parameter values can land at very different recall levels. Always
  measure both libraries across a parameter sweep and compare qps at
  matched recall. See `scripts/nndescent_jl_pareto.jl` for the pattern.

## Open work

### HNSW

- **Lock-free adjacency + visited-buffer pool (review ROI before
  committing).** Two pieces that together would close most of the
  remaining threading gap to hnswlib (estimated multi-thread build
  3.96s → ~2.0s, multi-thread query 16k → ~30-40k QPS at SIFT-128
  n=50k — these are old projections, re-measure baseline before
  committing). Slab adjacency: replace `Vector{Vector{Int}}` with
  `Matrix{Int}` + atomic degree counters, eliminate per-node
  `ReentrantLock`. Visited pool: lift `BatchQueryScratch`-style buffers
  from per-batch to index-lifetime ownership, atomic-stack pool.

  **Goal/rationale:** the only remaining big lever for HNSW. Won't
  reach hnswlib parity (C++ vs Julia constant factors cap us at ~1.4×
  behind, not 1.0×) but closes most of the gap. Cost ~2-3 days.
  Re-evaluate after thesis whether library publishability needs the
  extra speed-up or whether "competitive but Julia-native" is
  sufficient.

- **Pool the `BestCandidatesHeap` backing buffer for build.** Per-call
  heap allocation in `_search_layer` is the dominant remaining
  single-threaded build alloc (~36 KB/call at ef_c=200). Same coupling
  tradeoff as `visit_stamps` (build-only, single-threaded). Estimated
  10-20% single-thread build speedup. Smaller win than the threaded
  build work above.

### NN-Descent

- **Optional `bounded_candidates` / `max_candidates` knob for query.**
  Investigation (see git log) showed MANN's qps gap to NND.jl at fixed
  parameters is a recall/speed tradeoff, not a perf bug: MANN's
  candidates queue is unbounded (delivering higher recall) while NND.jl
  bounds at `max_candidates`. At matched recall MANN is 1.7-2.7× faster
  across the range both can reach. Exposing a `bounded_candidates` mode
  would let users opt into NND.jl-style faster-but-lower-recall query.
  Default to current unbounded behaviour. Not load-bearing.

- **Adaptive `max_iterations`.** Currently hardcoded at 10; PyNNDescent
  uses `max(5, round(log2(n)))`. Cheap fix, plausibly small recall
  improvement at large n.

- **PyNNDescent as a quality oracle.** Head-to-head validation against
  PyNNDescent on identical inputs would catch creeping recall drops the
  current ≥ 0.90 / MRR ≥ 0.92 floor doesn't notice. Costs Python in the
  validation loop; defer until the ecosystem question (Distances.jl /
  Manifolds.jl / NN.jl) is decided.

### LSH

- **Tiny follow-ups (consider later if worthwhile).** Both are non-issues
  at thesis-typical sizes; neither warrants a dedicated session.
  - Replace `shuffle!` + `resize!` in `query` with reservoir sampling
    when `candidate_cap << length(candidates)` (O(cap) instead of O(C)).
  - `_collect_candidates` calls `sizehint!(seen::BitSet, expected)`,
    misleading because BitSet storage scales with `max(id)` not element
    count. Either drop the line or revisit BitSet vs `Set{Int}` at large
    N (~1.25 MB/query at N=10M).

### KDTree

KDTree is reference-only for the thesis (the contribution lives in the
manifold-aware graph machinery, not the KD-tree). The first-pass
quickselect / in-place partition / query alloc cleanup landed; remaining
items below are textbook improvements (Friedman-Bentley-Finkel 1977) for
bringing the reference up to a respectable baseline.

**Second pass (do AFTER measuring whether it's still worthwhile):**

- **Leaf-bucket layer (leafsize ~16-32).** Stop recursion at a
  configurable bucket size; store the leaf as a `UnitRange` into the
  `indices` permutation, scan the bucket linearly in the query. Closes
  most of the remaining query gap at low/mid d (3-4× query speedup at
  d=8 expected). ~80-120 LOC across `types.jl`/`builder.jl`/`query.jl`.
  Additive — keep `KDTreeIndex` exported fields stable, internal layout
  change only.

- **Reorder data for leaf-contiguous memory.** Build a permuted
  `Matrix{T}` so leaf scans become sequential column reads instead of
  random `data[:, indices[i]]` lookups. 1.3-1.6× query speedup *on top
  of* the leaf-bucket change. ~30 LOC. **Only meaningful after the
  leaf-bucket change** — without leaves there's nothing to reorder.
  Make it opt-out so callers passing huge matrices can skip the copy.

**Lower-priority follow-ups:**

- **Incremental rolling-bound distance during query.** Standard
  Friedman-Bentley-Finkel 1977 trick: maintain the squared distance from
  `q` to the active cell; when descending into the far child, update
  only the changed axis's contribution. Tighter pruning than
  `abs(q_val - split_value)`. 1.2-1.4× query speedup expected at
  moderate d. ~40 LOC, gate on `default_distance` (only correct for
  additive Minkowski-like metrics). Probably not worth doing if the
  leaf-bucket change closes most of the query gap.

- **Hyperrectangle width tracking** as an *alternative* axis selector.
  Maintain `mins`/`maxes` down recursion and pick `argmax(maxes - mins)`
  instead of rescanning `indices` over `d` axes per node. Faster, but
  the resulting axis choice is not identical to `:variance`'s output
  (it's the cell width, not the data width). **Hesitant — do not adopt
  purely for performance.** The `:variance` axis selector is a
  thesis-relevant degree of freedom (it interacts with manifold-aware
  embeddings); changing the default for a constant-factor speed win is
  the wrong trade. If pursued, expose as a new `axis_selector = :bbox`
  and keep `:variance` as default.

**Anti-recommendations (do not do these):**

- Implicit binary heap layout (`getleft(i)=2i, getright(i)=2i+1`) for
  the node array. Saves one Int per node at the cost of significant
  bit-twiddling complexity.
- `SVector`-based hyperrectangle plumbing — forces the entire pipeline
  to be parametric on `D`, harder to read, more recompilation.
- Multiple coupled flags (`reorder` / `storedata` / `reorderbuffer`).
- Threaded build — the gap is closeable single-threaded; threading
  obscures the algorithmic story we'd want to tell.
- Metric-specific kernel branches (Chebyshev, etc.) — couples the tree
  to metrics we don't ship.

### Geodesic / graph

- **`pushfirst!(path, current)` in `_reconstruct_path`.**
  `src/geodesic/geodesic_model.jl:288`. O(n) per call → O(n²)
  reconstruction. Use `push!` then `reverse!` once.

- **`GreedySolver` complexity claim wrong.**
  `src/graphs/refinement/solvers.jl:459` documents O(k² log k); actual
  is O(k⁴) worst case. Either fix the algorithm (heap of edges) or fix
  the docstring.

- **`_fit_geometries(::ShareSimilarTangents)` boxing.** Both sharing
  variants use `Vector{Any}` because they fill out of order with
  `nothing` placeholders. Could tighten to `Vector{Union{Nothing, G}}`
  once `G` is known after the first fit. Low-priority polish.

### Latent type-narrowing in kmeans (multilevel-build path)

`KMeansTransform.fit!`, `init_random`, `init_kmeans_plus_plus`,
`pairwise_distances!`, and the kmeans `lloyd!` driver all type-restrict
to `::Matrix{T}` rather than `::AbstractMatrix{T}`. The narrow
signatures are intentional: `pairwise_euclidean!` /
`pairwise_sqeuclidean!` use BLAS `mul!`, which requires `StridedArray`
for GEMM dispatch — a non-strided gather `view(X, :, ::Vector{Int})`
silently falls to a slow generic path.

`_materialize_partition` (in `src/indices/multilevel/builder.jl`)
copies SubArrays into a fresh `Matrix` at the partition boundary so
this never bites in practice. **Goal/rationale:** the constraint is
load-bearing but easy to trip on if someone changes the partition
path. Worth either (a) leaving as-is and documenting the contract on
the kmeans signatures, or (b) widening to `AbstractMatrix` and
explicitly handling the strided/non-strided fork inside
`pairwise_distances!`. Defer until there's a real need.

### New indices to consider

- **`RPTreeIndex` as a standalone index.** RP-tree primitives already
  exist in the codebase (`src/indices/nndescent/rptree_init.jl`:
  `build_rptree`, `build_rptree_forest`, `leaf_members`) and are used as
  an opt-in init for NN-Descent. Promoting them to a top-level
  `RPTreeIndex` would be cheap (~50 LOC of wrapping struct +
  `build_index` / `query` methods) and covers the high-d regime where
  KD-tree degrades (>~50 dims) — RP-trees pick random hyperplanes and
  avoid the axis-aligned curse-of-dimensionality. Modest novel
  contribution in the Julia ecosystem: `NearestNeighbors.jl` ships
  KDTree/BallTree/BruteTree but not RP-tree forest; the only Julia
  RP-tree implementation today is buried inside
  `NearestNeighborDescent.jl`'s init. When this lands, move the
  primitives from `src/indices/nndescent/` to `src/utils/rptree.jl` so
  NN-Descent and `RPTreeIndex` share them.

### Benchmark harness fairness (`benchmarking/`)

Cross-library numbers from the general harness are now defensible after
the JIT-warmup, thread-control, data-marshalling, and native-batch-query
fixes (see git log for `d19aaad` and `f9765dc`). Two fairness gaps
remain:

- **Single-shot timing produces unstable per-algorithm QPS for noisy
  query variants.** At small batch sizes, between-query variance
  dominates the timed window; reproductions on pruned-deferred NN-Descent
  span ~2500-13500 qps across runs. **Fix shape:** `--reps N` CLI flag
  (default 1, preserves current behaviour for development sweeps);
  thesis-grade runs use `--reps 3` or `--reps 5` with median + IQR
  reporting. Force `gc.collect()` and `jl.GC.gc()` between reps and
  between algorithms.

- **Cross-library parameter sets are not automatically apples-to-apples.**
  Each algorithm in `benchmarking/configs/*.yaml` gets its own params
  block; some parameters mean the same thing across libraries (`k`,
  `max_iterations`), others look similar but dial recall at different
  rates (`ef_search` vs `max_candidates` — see "Apples-to-apples" in
  Working principles). The harness has no automated check that parameter
  sets correspond for cross-library pairs. **Fix shape:** a "comparable
  groups" YAML section that names which configs are meant to be
  cross-library apples-to-apples, plus either a check that shared
  parameters match or a comment block in each config explaining the
  equivalence.

**Lower-priority:**

- Query result conversion (Julia → 0-indexed Python ids) charged to
  Julia only — inside the `query_batch` timed region.
  `manifoldann.py:78-86,134-156`.
- pynndescent build (numba JIT) not warmed at the actual config — the
  8-query warmup helps the query path, not the BUILD path. First
  `NNDescent(...)` call still pays numba JIT inside the timed `fit()`.
  Currently moot (PyNNDescent excluded from configs); becomes relevant
  once enabled.
- No memory / allocation reporting — only wall time. Thesis claims
  about index footprint are not backed by harness output.
- KDTree query falls through to a Python loop in
  `manifoldann.py:324-326`. Now partially addressed by the
  AbstractANNIndex generic batch query (KDTree gets threaded batch via
  Julia-side dispatch). Remaining gap to SciPy's vectorised
  `cKDTree.query` is a Julia-side algorithm question, not a harness
  issue.

**Two ann-benchmarks-style additions worth folding in:**

- **Re-enable PyNNDescent in the benchmark runner.** Wrapper exists at
  `benchmarking/benchmarking/wrappers/pynndescent.py` but excluded from
  `benchmarking/configs/algorithms.yaml` with a stale comment about
  Python <3.10. Recent install on Python 3.13.5 succeeded; drop the
  comment and re-enable.

- **Enable NearestNeighbors.jl in configs.** A wrapper for
  `NearestNeighbors.KDTree` exists at `julia_external.py` and is
  registered, but not enabled in any config. No BallTree wrapper at all.

### Repository hygiene

- **`scripts/` directory cleanup.** Currently 50+ files, mixed bag from
  multiple thesis workstreams (composite-shortcut research, ORC/curvature,
  geodesic, perf scripts). Worth restructuring into subdirectories by
  purpose — defer concrete moves until the user can triage which
  research scripts are still active. Suggested structure:
  - `scripts/perf/` — `hnsw_*`, `nndescent_*`, `kdtree_*`, `distance_*`
  - `scripts/research/` — composite-shortcut, ORC, geodesic experimental
    scripts (move to thesis-code repo if/when `benchmarking/` migrates)
  - `scripts/diagnostics/` — `diag_eff_eps.jl`, `diagnose_sinkhorn.jl`
  - `scripts/archive/` — already exists; sweep stale items in here
  - Top-level: `README.md`, `run_*.sh` orchestration scripts only
  - `scripts/__pycache__/` should be in `.gitignore`.

- **`benchmarking/` migration to a sibling thesis-code repo** (per
  `CLAUDE.md`'s separation principle). Strategic move not blocking
  immediate work. Pre-requisite: harness fairness fixes above. After
  migration, the package's perf surface lives in `scripts/perf/`; the
  thesis-coupled multi-algorithm sweep harness lives in the sibling
  repo.

### Architecture / idiomatic-Julia review backlog

Items surfaced by the read-only architecture review. Mechanical cleanups,
none gating.

- **Doc bloat.** Multilevel docstrings duplicate the same IVF example
  3-4× across `multilevel/{multilevel_index,multilevel,transformed,routing}.jl`.
  Export list in `src/ManifoldANN.jl` is ~185 names with ~20-30
  internal-only. Worth a focused doc/exports pass.

- **Submodule-style imports without submodules.**
  `src/indices/multilevel/multilevel_index.jl:47-52` and friends use
  `using ...ManifoldANN: ...` (3 dots). Works only because Julia
  tolerates the extra dot; if a real submodule is ever introduced, these
  silently change meaning. Cosmetic but gnarly.

- **TODO_cleanup.md → durable principles split.** This file's "Working
  principles" and "Anti-patterns" sections are durable invariants; the
  rest is forward-looking. Reviewer's recommendation: move the durable
  parts to `AGENTS.md` or `docs/architecture.md`, keep this file purely
  forward-looking. Not done yet.

### Areas not reviewed

The independent code review skipped: `src/indices/multilevel/*`
(IVF-HNSW), `src/transforms/kmeans/*`, `src/preprocessing/*`,
`src/geodesic/refinement.jl`, `src/geometry/neighborhood.jl`,
`src/geometry/criteria.jl`, ORC `EdgeNeighborhoodView` construction
(`graphs/refinement/{types,filtering,effective_epsilon_policy}.jl`).
Worth a second pass before claiming the package has been
comprehensively reviewed.

## Strategic decisions outstanding

### Julia-ecosystem-native vs self-contained

Three workstreams point in the same direction and should be decided
together, not as discrete fixes:

- **Distances.jl as the metric provider.** Replace `default_distance`
  etc. with thin aliases over `Euclidean()` / `SqEuclidean()` / etc.
  Wire `Distances.pairwise!` into bruteforce builder/query and k-means
  Lloyd kernels — that's where the BLAS3 win materialises (cross-term
  identity dispatches to GEMM). Per-call `evaluate(metric, x, y)` is
  **no** SIMD win over our current `@simd` kernels — only bulk
  pairwise/colwise pays. Half a day for the alias swap; 2-3 days for
  `pairwise!` integration with benchmarks. Risk: BLAS-backed pairwise
  can produce small negatives from cancellation; bit-for-bit results
  may differ from `@simd` reduction and could perturb tie-breaking in
  unit tests.

- **Manifolds.jl as the manifold/sampler provider.** Wire as a sampler
  + ground-truth-distance backend. Adapter: `sample_manifold(M, n) →
  (ambient, intrinsic)`, ~30 LOC per manifold, built on `rand(M)` /
  `embed(M, p) |> vec` / `distance(M, p, q)`. Concrete manifolds worth
  adding for the experimental story: $S^2$, $H^2$, $\mathrm{Gr}(4,2)$,
  SPD $P(3)$ (vech-flattened), $SO(3)$. Friction: typed-point
  representations vary (SPD is `Symmetric{Matrix}`, Grassmann is
  `Matrix`, etc.) — the ambientize layer must flatten via
  `embed(M,p) |> vec`. EmbeddedTorus has no closed-form geodesic in
  Manifolds.jl either; the existing torus would still need
  grid-Dijkstra. Local PCA tangent estimation degrades when ambient
  dim ≫ intrinsic dim or when the true tangent space is an affine
  subspace of a non-linear variety (SPD): the Manifolds.jl-exact-tangent
  vs local-PCA comparison would actually strengthen the empirical
  narrative. Dependency cost: Manifolds.jl + ManifoldsBase.jl +
  RecursiveArrayTools + StaticArrays + Distributions + ManifoldDiff
  (~3-5s load time). Effort: 1-2 weeks.

- **NearestNeighbors.jl as an alternative ANN backend.** Mature,
  threaded, gives us BallTree (useful for higher-d regimes where
  KD-tree degrades). Replacing our KD-tree for parity is not worth it;
  adding a `NearestNeighborsBallTree` index wrapper would be cheap and
  additive.

The thesis benefits from the current self-contained stance (clear
scope, fewer moving parts). A published library benefits from
ecosystem-native (less code to maintain, broader compatibility, easier
for others to extend). **Decide direction first**, then sequence the
three as one coherent effort.

## Anti-patterns (don't re-explore these)

One-liners pointing at full context in commit history.

- **NN-Descent dict-based dedup variant**: 18-35% slower than baseline
  at k∈{15,30,50}. Set/hash overhead exceeds the cheap O(k) scan at
  typical k. See `d9bab8a` for the simpler simplification that did land.

- **NN-Descent `_update_neighbor_dist!` "asymptotic fix"**: the
  catastrophic branch is unreachable on deterministic distances (same
  `(a,b)` always yields the same distance, so `dist < existing.dist`
  never fires). The perf TODO that flagged it was based on an
  unreachable path. See `d9bab8a`.

- **RP-tree initialization for NN-Descent**: did NOT close the build
  gap. All three implementation variants tested were 1.6-10× *slower*
  than random init. Reason: with bidirectional random init, NN-Descent
  converges in 1-2 iterations on this codebase's
  `convergence_threshold=0.001` default — no iterations to save. Did
  improve recall (n=5000 d=32 k=15: 0.92 → 0.96, +3.6pp). Retained as
  opt-in via `init=:rptree`. See `05b91d3`.

- **Threading rework via `@sync` + `@spawn` over chunked ranges**:
  tried as a fix for "30% idle" profile result, ended 6-9% slower. The
  "idle" was task-switching, not thread starvation. Per-task alloc +
  spawn/sync cost exceeded the load-imbalance savings at our work
  granularity.

- **Edge-weight threading deferred deliberately**: edge weights are
  ~10% of graph build time (PCA fit dominates), so Amdahl caps the win
  at ~10%, AND `compute_edge_weight` is a public extension point —
  silently threading the outer loop would impose an undocumented
  thread-safety contract on third-party `AbstractEdgeWeight` /
  `AbstractLocalGeometry` implementations. Revisit only after
  formalising the thread-safety contract for those extension points.
  If pursued, expose as opt-in.

- **`ShareSimilarTangents` threading**: semantically serial. Each
  iteration's share/fit decision depends on which earlier nodes were
  fitted vs shared. Threading would lose the sharing structure or
  produce schedule-dependent outputs. Leave serial by design.

- **NN-Descent batch-query data_cache experiment** (mid-session,
  reverted): hypothesis was that materializing data into
  `Vector{Vector{T}}` at the end of `build_index` would speed up query
  inner loops by avoiding per-call SubArray construction. Microbench
  showed 3.76× faster per distance call; full integrated query path
  showed -37% to -50% qps regression. The refactor (untyped `columns`
  helper passed through hot loop) defeated specialization more than the
  layout change saved. Trust integrated benches over microbenches in
  this codebase.

- **Type-parameterise KDTree's `distance::Function`**: the change is
  idiomatic and landed (matches every other index), but the audit's
  predicted 5-25% query speedup did not materialise. The compiler
  already devirtualises via single-call-site specialisation on the
  default. Same applies to multilevel `::Function` kwargs (also
  landed). The cleanup is worth doing for consistency, not for measured
  perf.
