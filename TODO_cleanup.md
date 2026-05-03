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

- **Pool the heap backing buffers in `_search_layer` (deferred).** The
  pending `NeighborMinHeap` accumulates ~ef·M candidates per call (~50 KB
  peak at ef_c=200) and reallocates as it grows; this is the dominant
  remaining build alloc once the slab + visited-pool work landed. Profile
  on n=20k d=128 attributes ~9% of total build time to `_growend!` on the
  heap vector (samples in `array.jl:1131`/`1148`/`1289`), so an upper
  bound on the win is ~10%. Skipped for now: closing it requires changing
  `_search_layer`'s ownership model (it currently returns `state.best.data`
  to the caller — pooling means an explicit copy or a documented
  "valid-until-next-acquire" contract on the hot path). Worth revisiting
  if we want to close the remaining ~1.3× gap to hnswlib at build time;
  10% on a 1.3× gap is meaningful.

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

LSH is a thesis baseline and is delivering ~85% recall at typical configs;
no further perf work is load-bearing. Items below are tiny polish; defer
unless LSH becomes a more central thesis comparison.

- **Reservoir sampling for `candidate_cap`.** `query` currently uses
  `shuffle!` + `resize!` on the full candidate list — O(C) when O(cap)
  would do. Non-issue at thesis-typical candidate sizes.
- **BitSet vs `Set{Int}` at large N.** `_collect_candidates!`'s
  `sizehint!(seen::BitSet, expected)` is misleading because BitSet
  storage scales with `max(id)` not element count. With per-task pool
  the BitSet now has a fixed `n_points/8`-byte high-water mark per
  task; correct as-is, but the comment-or-swap discussion remains.
- **CSR (offsets + flat ids) layout for tables.** Build-once,
  query-many is the actual LSH use case; CSR would give cache-friendly
  bucket scans. Blocked by `insert!` support — would require dropping
  it (LSH `insert!` exists today and matches the `supports_mutation`
  trait, but has no production callers). Modest win (~10-20% qps),
  not load-bearing.
- **`mean_bucket_size` formula is uniform-distribution.** Real LSH
  buckets are skewed by design; the sizehint underestimates in skewed
  regimes. Capped at `n_points` so no catastrophe. Worth a comment.
- **`hash_factory` extension-point contract.** 689974b auto-injects
  `T = eltype(data)` for the built-in factories (`make_random_hyperplane_hash`,
  `make_binning_hash`); user-supplied factories receive `hash_kwargs`
  verbatim (743f57b). If we ever document the hash_factory extension
  point publicly, the contract should be either "your factory must
  accept `T`" or "you must pre-bind `T` via a closure." Decide when
  surfacing it.

- **Zero-norm cosine semantics changed in 0834a73.** Old hand-rolled
  `cosine_distance` returned `Inf` on zero-norm inputs (sort them last
  in priority queues); `Distances.CosineDist()` returns `NaN`. No
  current callsite is affected (no zero vectors in test data, no code
  reads ordering of NaN distances). If a user reports
  `RandomHyperplaneHash` results misbehaving on a dataset with zero
  vectors, this is the difference. Either revert to a thin wrapper
  enforcing `Inf` semantics, or document the precondition (no
  zero-norm inputs).

### KDTree

KDTree is reference-only for the thesis (the contribution lives in the
manifold-aware graph machinery, not the KD-tree). Leaf-bucket layer +
router-only internal nodes landed (5bae18d / 8aa89ff); distance-metric
safety gate landed (f4d1acd / 1a9b0b4) excluding non-componentwise-monotone
metrics from the build API.

**Latent correctness issue worth fixing properly:**

- **Pruning is in mixed units.** `query.jl:60` compares linear
  `axis_distance = abs(q_val - split_value)` against `worst`, but for
  squared metrics `worst` is in squared units. We currently work
  around this by excluding `SqEuclidean` from the safe-metric list.
  The clean fix is the **incremental rolling-bound distance**
  (Friedman-Bentley-Finkel 1977): maintain `q-to-cell` squared
  distance through descent, update one axis's contribution at the
  far-child gate. Compares like-for-like; would unblock SqEuclidean
  re-admission to the safe list. ~40 LOC, gated on additive metrics.

**Lower-priority follow-ups:**

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

- **Reorder data for leaf-contiguous memory.** Would give 1.3-1.6× query
  speedup on top of leaf-buckets but is the canonical NN.jl-shaped
  feature kit; deliberately not pursued — package is meant to be a
  reference baseline, not a polished alternative to NearestNeighbors.jl.
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

### Multilevel / IVF follow-ups

A through D landed this session along with a brutal-critic pass. Mostly
clean; residuals worth knowing about:

- **Tie-break ordering changed in inverted-loop batch dispatch.** Old
  single-vector path appended results in parent-routing order; new
  matrix path (d24a7c3) appends in `child_idx` order. For tied
  distances, top-k selection picks different elements. Δrecall on 1000
  GloVe-100 queries: +0.0001 — well within noise. Both orderings are
  algorithmically valid; the difference is a deterministic-merge
  detail. No action; document only if a user reports a recall flake.

- **Empty-children edge case.** If a query routes to zero children
  (`probe_indices` empty), `merge_results(SimpleMerge(), [], k)` is
  called with an empty list. Currently works (returns empty Vector) but
  the contract isn't explicitly tested. Worth a defensive testset if
  this comes up.

- **Per-batch `queries_per_child` allocates `n_children` empty `Int[]`
  vectors** even for empty cells. At nlist=128 that's 128 empties per
  batch — small but real. CSR-style flat layout (counts pass + offsets
  + flat array) would replace the Vector-of-Vectors with two
  contiguous Int arrays. Bounded value; defer.

- **Doc bloat in multilevel.** Multilevel docstrings duplicate the same
  IVF example 3-4× across `multilevel/{multilevel_index,multilevel,
  transformed,routing}.jl`. Worth a focused doc consolidation pass.

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

- **Alternative `AbstractRPSplitter` implementations.** The splitter
  trait landed in 6e9aa3a with `TwoPointSplitter` as the default. Worth
  exploring as experimentation surface (not thesis-load-bearing):
  - **PCA-aligned splits**: hyperplane = top principal component of the
    points in the current node. Tighter splits in low intrinsic
    dimension; pays a per-node SVD cost. Good fit for the manifold-aware
    framing of the thesis.
  - **Mondrian splits**: exponential-clock-driven random axis-aligned
    splits with a lifetime parameter (Lakshminarayanan et al.). The
    splitter trait alone fits, but full Mondrian needs split-time +
    bounding-box state per node and an online `insert!` API — a separate
    `MondrianTreeIndex`, not a splitter swap. The splitter abstraction
    is offline-only; if Mondrian ever lands, treat it as a sibling
    index, not an RPTree variant.

- **Shared "binary partition tree with leaf buckets" abstraction
  (deferred).** KDTree and RPTree share a meta-pattern: recursive binary
  partitioning of a point set with router-internal-nodes and
  leaf-buckets. The split content differs (axis-aligned vs hyperplane),
  the storage layout differs, the query path differs (KDTree prunes,
  RPTree doesn't), so the shared skeleton is shallow (~20-30 LOC of
  recursion) and the per-tree specialisation is essential. Don't factor
  with only two instances; the Julia idiom (`NearestNeighbors.jl`:
  KDTree/BallTree/BruteTree as separate concrete types under a thin
  `NNTree` protocol) supports this. **Trigger to revisit**: a third
  tree-style index (BallTree, M-tree, vp-tree) lands. At that point the
  pattern is real and the factoring pays back.

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

- **Export list bloat.** `src/ManifoldANN.jl` is ~185 names with ~20-30
  internal-only. Worth a focused exports audit (multilevel doc bloat
  noted under the multilevel section).

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

The independent code review skipped: `src/transforms/kmeans/*`
(but kmeans got the subsample-for-training pass — partial review),
`src/preprocessing/*`, `src/geodesic/refinement.jl`,
`src/geometry/neighborhood.jl`, `src/geometry/criteria.jl`, ORC
`EdgeNeighborhoodView` construction
(`graphs/refinement/{types,filtering,effective_epsilon_policy}.jl`).
Worth a second pass before claiming the package has been
comprehensively reviewed. Multilevel/IVF was reviewed in this session
across A/B/C/D and the brutal-critic pass — see "Multilevel / IVF
follow-ups" below for residual items.

### ORC / refinement / tangent-sharing test coverage

`ShareSimilarTangents` and its helpers (`_fit_geometries`,
`_find_shareable_geometry`, `_fit_geometries_with_candidates`,
`_find_shareable_geometry_candidates`) have zero unit-test coverage.
A broken constructor in those functions slipped through `Pkg.test()`
and was caught only when an unrelated bench script invoked the path.
Tangent-sharing is exercised via thesis experiments but not via the
test suite. Worth a focused coverage pass — at minimum, smoke tests
that the four functions execute without error on a small graph.
Same likely true for other ORC / refinement code (curvature solvers,
`EdgeNeighborhoodView` construction); audit coverage broadly, not
just the one site that bit us.

## Strategic decisions outstanding

### Julia-ecosystem-native vs self-contained

Three workstreams point in the same direction and should be decided
together, not as discrete fixes:

- **Distances.jl as the metric provider — alias layer landed (0834a73).**
  `default_distance` / `default_squared_distance` / `euclidean_distance`
  / `cosine_distance` are now thin aliases over Distances.jl types;
  hand-rolled SIMD kernels deleted. `squared_cosine_distance` kept as a
  function (genuine semantic mismatch — `2·(1-cos_sim)` ≠ anything in
  Distances.jl). Microbench confirmed per-pair perf parity. Remaining
  work: wire `Distances.pairwise!` into bruteforce builder/query for the
  BLAS3 batch win (kmeans already uses it). 2-3 days for that
  integration with benchmarks. Risk: BLAS-backed pairwise can produce
  small negatives from cancellation; bit-for-bit results may differ
  from `@simd` reduction and could perturb tie-breaking in unit tests.

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
