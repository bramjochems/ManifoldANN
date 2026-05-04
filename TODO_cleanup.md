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
  correctness gate (`test/unit/` + `test/regression/`, both must-pass);
  perf is comparative measurement, lives in `scripts/perf/`, and is run
  by hand around changes that might move it. Conflating them gives you a
  CI gate that's both architecture-dependent (perf bands fail on
  unrelated machines) and recall-blind (regression gets buried under
  flaky timing). Keep them separate.

- **`test/unit/` vs `test/regression/` split.** Both run on every
  `Pkg.test()`. `unit/` = single-function-or-cluster, milliseconds; runs
  on every save. `regression/` = end-to-end pipeline outputs (snapshot
  hashes, recall floors, eltype propagation across stages). Both
  correctness, both must-pass — the only real difference is granularity.
  Put new tests in whichever fits; don't gate on opt-in flags.

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
  must come from a focused fair-compare script in `scripts/perf/` (e.g.
  `scripts/perf/hnsw_fair_compare.py`, `scripts/perf/nndescent_jl_pareto.jl`,
  `scripts/perf/kdtree_fair_compare.jl`).

- **Apples-to-apples library comparisons go on the recall-vs-qps Pareto
  curve, not at fixed parameter values.** Different libraries' search
  parameters (e.g. `ef_search` vs `max_candidates`) bound different data
  structures and dial recall at different rates. Numerically-equal
  parameter values can land at very different recall levels. Always
  measure both libraries across a parameter sweep and compare qps at
  matched recall. See `scripts/perf/nndescent_jl_pareto.jl` for the pattern.

## Open work

### NN-Descent

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
safety gate landed (f4d1acd / 1a9b0b4) excluding non-additive metrics
from the build API. The query path now uses the **Friedman-Bentley-
Finkel 1977 incremental rolling-bound prune** for Euclidean,
SqEuclidean, Cityblock, and Minkowski — unit-consistent compare,
SqEuclidean is back on the safe list. `WeightedEuclidean`,
`WeightedMinkowski`, and `WeightedCityblock` are also on the rolling-bound
path (per-axis contribution `w[axis] * excess^p`); the legacy
`axis_distance <= worst` prune over-pruned them when weights < 1 — fixed.
Chebyshev remains on the legacy prune (correct there: max-reduce, not
sum-reduce, so the rolling-bound code path doesn't apply).

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

- **Mondrian-style tree.** Exponential-clock-driven random axis-aligned
  splits with a lifetime parameter (Lakshminarayanan et al.). The
  `AbstractRPSplitter` trait alone fits, but full Mondrian needs
  split-time + bounding-box state per node and an online `insert!` API —
  a separate `MondrianTreeIndex`, not a splitter swap. The splitter
  abstraction is offline-only; if Mondrian ever lands, treat it as a
  sibling index, not an RPTree variant.

- **Generalize forest as `BPTForestIndex` keyed on a randomization marker
  trait.** Currently `RPTreeForestIndex` and `PCATreeForestIndex` are
  separate concrete types; both wrap "build N parallel BPT trees with
  `spawn_child_rngs` + union leaf buckets at query." A generic forest
  gated on a marker trait that splitters opt into ("yes, I genuinely
  consume rng across calls") would absorb both — and would
  compile-time-prevent forests over deterministic splitters (which
  produce N identical trees). Trigger to revisit: a third
  randomized-tree index lands (Mondrian, randomized BallTree, or
  randomized VPTree).

- **Depth-adaptive PCA-then-RP meta-splitter.** Implementable as a
  meta-splitter that wraps a `PCASplitter` and an `AbstractRPSplitter`
  and dispatches in its own `bpt_split!` overload — without modifying
  the four trait types. Trigger: empirical evidence that PCA tightens
  early splits but RP wins at depth where node samples get too small
  for stable spectra. Defer until benchmarked.

- **Ball-tree and VP-tree as next instances of the shared BPT
  abstraction.** `src/utils/binary_partition_tree.jl` landed alongside
  `PCATreeIndex` and validates against the recursive-binary-partition
  + leaf-bucket meta-pattern. Ball-tree (centroid + radius router) and
  VP-tree (vantage point + median radius router) are the two natural
  next instances — each just supplies a `bpt_split!` and a per-tree
  router-payload struct. The descent rule will differ from PCA/KD/RP
  (radius-based pruning), so each tree keeps its own query path.

- **KDTree refactor onto the shared BPT helper (deferred).** RPTree
  was migrated onto BPT alongside the brutal-critic pass — BPT now has
  two consumers (PCA + RP) and earns its keep as a real abstraction.
  KDTree stays on its bespoke in-place `(lo, hi)` + `quickselect!`
  layout: BPT's contract requires fresh `Vector{Int}` per split, which
  KDTree deliberately avoids. Forcing KDTree onto BPT would require a
  Partitioner trait change; defer until Ball-tree or VP-tree forces
  the question.

### Benchmark harness fairness (`benchmarking/`)

The harness is now thesis-defensible for breadth comparisons after the
fairness pass (`--reps`, split-timing, build-warmup, memory tracking,
comparable-groups). Specific QPS claims in the thesis still come from
focused fair-compare scripts in `scripts/perf/`, not this harness.

What landed:

- `--reps N` with median + IQR reporting on QPS and recall. Full Python
  + Julia GC (`gc.collect()`, `jl.GC.gc()`, `jl.GC.gc(true)`) between
  reps AND between algorithms; one collection runs immediately before
  each timed window so a stale GC pause never gets charged to the rep.
- Build-path JIT warmup at the *actual* config (not a tiny stub) for
  Julia builders (HNSW, NN-Descent) and for PyNNDescent (numba JIT).
  Costs seconds once per (algo, dim).
- Query-path warmup uses the actual `k` and batch size, going through
  `query_batch_raw` so the id-conversion path JITs too.
- Result id-conversion (Julia 1-based → Python 0-based) moved out of
  the timed region: wrappers expose `query_batch_raw` (timed,
  algorithm-side only) and `finalize_batch_ids` (untimed, marshalling
  only) via the base-class contract. Updated in `manifoldann.py` and
  `julia_external.py`. ManifoldANN batch results now go via a
  Julia-side `Matrix{Int}` so the whole batch crosses the boundary as
  one numpy view.
- KDTree batch path no longer loops singletons from Python — the
  generic `AbstractANNIndex` matrix dispatch handles it threaded
  Julia-side.
- Per-library effective-thread-count check at run start (Julia
  `Threads.nthreads()`, `BLAS.get_num_threads()`,
  `faiss.omp_get_max_threads()`, `OMP_NUM_THREADS`,
  `JULIA_NUM_THREADS`). Catches silent oversubscription.
- Memory reporting: peak RSS delta during build and during query phases
  via `psutil`; new CSV columns `build_rss_delta_mb`,
  `query_rss_delta_mb_max`. Library-reported `memory_usage()` hook
  exposed on the wrapper base; populated nowhere yet (see open items).
- `comparable_groups:` YAML section in dataset configs names
  cross-library apples-to-apples sets. Harness validates members'
  recall ranges overlap (warns when spread > 0.10) and emits a
  recall-vs-qps Pareto-style CSV per group when `--save-output` is on.
  Initial groups in `fashion-mnist.yaml`: HNSW (MANN/HNSWlib/HNSW-jl),
  KDTree (MANN/SciPy/NN.jl), NN-Descent (MANN/NND.jl/PyNNDescent).
- PyNNDescent re-enabled in registry, `algorithms.yaml`, and
  `fashion-mnist.yaml` (Python 3.13.5 install works; the Python <3.10
  comment was stale).
- NearestNeighbors.jl KDTree enabled in `fashion-mnist.yaml` (the
  KDTree comparison group needs it). No BallTree wrapper exists or is
  planned.

Open / lower-priority:

- Pareto CSVs reflect a single parameter set per algorithm at run time;
  the genuine recall-vs-qps Pareto sweep belongs in focused scripts
  (`scripts/perf/hnsw_fair_compare.py`, etc.). Treat the harness CSV
  as the single-point head-to-head sanity check, not as the thesis
  Pareto.
- Comparable-groups validation uses point recall (or IQR if
  `--reps>1`), not a symbolic-parameter check on shared knobs (`M`,
  `ef_construction`, etc.). A symbolic check would catch a config
  typo that the recall-spread heuristic misses; deferred pending real
  false-positives.
- Library-reported `memory_usage()` is a no-op default on every
  wrapper; `index_mb` in the CSV is empty. Build/query peak-RSS delta
  is populated and is sufficient for thesis-grade footprint claims,
  but a true per-library index-footprint number would be more rigorous.
- Comparable groups defined for `fashion-mnist.yaml` only; replicate
  to the other dataset configs as they get used for cross-library
  claims.

### Repository hygiene

- **`benchmarking/` migration to a sibling thesis-code repo** (per
  `CLAUDE.md`'s separation principle). Strategic move not blocking
  immediate work. Pre-requisite: harness fairness fixes above. After
  migration, the package's perf surface lives in `scripts/perf/`; the
  thesis-coupled multi-algorithm sweep harness lives in the sibling
  repo. `scripts/thesis/` and `scripts/perf/` would migrate with it.

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

### ORC / refinement test coverage audit

Smoke coverage for `ShareSimilarTangents` and its helpers
(`_fit_geometries`, `_find_shareable_geometry`,
`_fit_geometries_with_candidates`, `_find_shareable_geometry_candidates`)
landed in `test/unit/graphs/share_similar_tangents_tests.jl`. The
broader audit is still open: other ORC / refinement code (curvature
solvers, `EdgeNeighborhoodView` construction) likely has similar
coverage gaps — audit broadly, not just the one site that bit us.

### ORC OT-solver follow-ups

- **`scripts/archive/benchmark_orc_comprehensive.jl` uses `randn` data**
  which defeats Sinkhorn convergence and routes Hungarian neighborhoods
  through pathological LP cases. Archived because the script was
  superseded; if it ever comes back, swap to a manifold sampler (swiss
  roll / sphere / torus) before drawing thesis numbers from it.

## Strategic decisions outstanding

### Julia-ecosystem-native vs self-contained

One workstream remaining (Distances.jl integration is closed — alias
layer landed in 0834a73; bruteforce `pairwise!` wiring investigated and
shelved, see Anti-patterns):

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

The thesis benefits from the current self-contained stance (clear
scope, fewer moving parts). A published library benefits from
ecosystem-native (less code to maintain, broader compatibility, easier
for others to extend). Manifolds.jl is the only ecosystem-integration
move still on the table; decide post-thesis.

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

- **NN-Descent adaptive `max_iterations = max(5, round(log2(n)))`**
  (PyNNDescent default): measured on fashion-mnist-784 at n∈{10k,60k}
  k=32, sweep ef∈{10,20,40,80,160}. n=10k: max_iter 10 vs 13 — recall
  delta ≤0.1pp at every ef. n=60k: max_iter 10 vs 16 — recall delta
  ≤0.5pp at every ef (signs split, within noise). Build time difference
  was noise (early convergence kicks in around iter ~6 on this
  codebase's `convergence_threshold=0.001` default — same reason RP-tree
  init didn't close the gap). Default left at 10. Sweep script:
  `scripts/perf/nndescent_max_iter_sweep.jl`.

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

- **`Distances.pairwise!` (BLAS3) batch path for `BruteForceIndex`**:
  investigated, did not win on the realistic deployment shape. With
  `JULIA_NUM_THREADS=4` (BLAS=8), the per-query threaded loop already
  saturates compute; a specialised batch method calling
  `Distances.pairwise!(metric, D, data, queries; dims=2)` once for the
  whole batch was 0.78-0.96× on Euclidean/SqEuclidean and only 1.07-1.17×
  on CosineDist at n=10k d=128 nq∈{100,1000} k=10. At n=20k mixed
  (Euclidean regresses 0.86×, Cosine wins 1.15-1.41×); only at n=50k
  does it win across the board (1.8-2.3×). Single-query also regresses
  (0.34-0.5×) — GEMV setup cost dominates at d=128. With
  `JULIA_NUM_THREADS=1` (BLAS=8), pairwise! wins everywhere — but
  realistic deployment is multi-threaded Julia. Bench:
  `scripts/archive/bruteforce_pairwise_bench.jl`. Don't re-explore unless the
  batch path becomes load-bearing at n≥50k specifically. The kmeans
  `pairwise_distances!` win is genuine because k-means hits k×n with
  k≪n on every Lloyd iteration; bruteforce hits n×nq with n,nq similar
  scale, where threading already eats the BLAS3 advantage.

- **Type-parameterise KDTree's `distance::Function`**: the change is
  idiomatic and landed (matches every other index), but the audit's
  predicted 5-25% query speedup did not materialise. The compiler
  already devirtualises via single-call-site specialisation on the
  default. Same applies to multilevel `::Function` kwargs (also
  landed). The cleanup is worth doing for consistency, not for measured
  perf.
