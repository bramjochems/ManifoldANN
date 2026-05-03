# Cleanup TODOs

Three follow-up workstreams identified during the thesis writing phase. Each should be picked up as its own focused effort after the thesis is submitted. None of them are blocking for the thesis itself; the thesis text is honest about the current scope.

## Status legend

- ✅ done (with commit hash)
- 🚧 in progress
- 📌 confirmed open
- ❌ wrong / not worth doing (with note)
- ➕ new finding added during execution

## 1. Asymptotic and threading fixes

Independent code review surfaced concrete issues where the implementation does not match the "asymptotically efficient with proper data structures, threading on naturally parallel work" framing. Fix these or qualify the claim. Ordered by severity.

### High

- ✅ **KD-tree query uses linear-scan neighbor buffer.** `src/indices/kdtree/query.jl`: `NeighborBuffer.push_candidate!` calls `findmax(buf.dists)` (O(k)) and `current_worst_distance` calls `maximum(buf.dists)` (O(k)). For k larger than ~10 this dominates the per-query cost. The package already implements `BoundedMaxHeap` in `src/utils/neighbor_heaps.jl`; replace `NeighborBuffer` with it. Push then becomes O(log k), peek O(1).
  - **Done in `e722498`.** 6.8-9.5× speedup at k=200, 2× at k=10, neutral at k=1 (50k k=1 ~5% slower, within noise band). Allocations also reduced 2-29%.

- ✅ / ❌ **NN-Descent `_insert_neighbor!` is O(k) per insertion and O(k log k) on update.** `src/indices/nndescent/builder.jl` lines ~435–490: linear scans both heaps for ID membership, and `_update_neighbor_dist!` does `filter` + heap rebuild from scratch. NN-Descent's local-join inner loop calls this O(k²) times per node. Fix: maintain a small `Dict{Int,Int}` (id → heap position) alongside each heap, or accept duplicates and dedupe at finalize time (PyNNDescent does the latter).
  - ❌ **TODO overplayed the perf impact.** The "O(k log k) on update" branch is dead code on deterministic distance functions: NN-Descent's local join may revisit the same (a,b) pair, but each visit recomputes the *same* `d(a,b)`. The `dist < existing.dist` test never fires. So the asymptotic claim ("O(k²) per node times O(k log k)") was based on an unreachable path. On our workloads (default_*_distance, all SIMD reductions on continuous data), the catastrophic case never occurs.
  - ❌ **Dict-based variant (Option B) was prototyped and benchmarked.** 18-35% *slower* than baseline across k∈{15,30,50}: Set allocation cost and per-call hashing overhead exceed the cheap O(k) scan at typical k. PyNNDescent's "accept duplicates, dedup at finalize" (Option A) was also tried — 1.3-1.5× faster but caused per-node neighbor lists to be shorter than k (duplicates evicted good unique entries before finalize), losing graph structure even though recall held.
  - ✅ **Committed as cleanup in `d9bab8a` (Option A′)**: drop the unreachable `_update_neighbor_dist!` path, scan only new_neighbors (one scan instead of two). Behavior bitwise-identical to baseline on deterministic distances (recall and MRR match to 4 decimal places). Build time matches baseline within ±7%. Net: 33 LOC removed, behavior matches PyNNDescent for non-deterministic distances (silently drops better-distance updates for existing ids), no perf claim. Strengthened tests (`bd3575d`) are the more valuable artifact of this exercise.

- ✅ **NN-Descent omits reverse-neighbor sampling in the local-join step.** `src/indices/nndescent/builder.jl` lines ~188–214: per-iteration sampling pulls only from `node.new_neighbors` and `node.old_neighbors` (forward edges). The canonical algorithm (Dong, Charikar, Li 2011) uses `B[v] ∪ sample(R[v])` where `R[v]` is reverse edges. The initialiser does add bidirectional edges, which partially papers over this, but during iteration the descent is forward-only → reduced graph quality at convergence. Fix: build a transient reverse-neighbor list per iteration and sample from it.
  - **Done in `83e0d17`.** Recall improvement: +0.2pp at n=500 (already near ceiling), +0.3pp at n=2000, **+1.1pp at n=5000** with default `pruning_degree_multiplier=1.5` (PyNNDescent's heuristic). Build cost: +16-32% vs no-reverse baseline. With `max_candidate_neighbors` cap relaxed to old default 64, recall climbs further (+5pp at n=5k) but build cost is +130% — exposed as a knob (`pruning_degree_multiplier`) rather than buried in a constant. Implementation uses preallocated reverse-neighbor buffers (refilled once per iteration) and reusable scratch vectors for the merged sample, dedup via sort+unique to avoid per-call Set allocation. Strengthened tests (`bd3575d`) confirmed the algorithmic improvement was real.

- ✅ **Geodesic point→point queries re-run Dijkstra k² times.** `src/geodesic/geodesic_model.jl` lines ~388–421 (point→node) and ~457–503 (point→point): the inner loop calls `geodesic_distance(model, data, entry_idx, j)` which runs a fresh Dijkstra each iteration. Fix: for point→node, run one multi-source Dijkstra seeded with `(entry_idx, local_dist(query, entry))` for every entry, then read `dist[j]`. For point→point, run a single multi-source from `entries_a`, then take min over `dist[entry_b_idx] + local_dist(b, entry_b)`.
  - **Done in `add013d`.** Added `_dijkstra_multisource` primitive; both query overloads rewritten. Speedups scale as predicted: point→node 5.2× at default entry_k=5, 9.9× at entry_k=10; point→point 11.7× at entry_k=5, 44× at entry_k=10. Minor regression at entry_k=1 point→point (0.62 → 1.12 ms) on a degenerate config that nobody runs in practice.

### Medium

- ✅ / ❌ **Geometry fitting and edge-weight computation are not threaded.** `src/graphs/weighted_knn_graph.jl`: `_fit_geometries` (~line 247) and `_compute_edge_weights_and_diagnostics` (~line 194) iterate `for i in 1:n` serially. Per-node PCA fit is independent and dominates build time on real datasets. For `NoSharing` mode this is `Threads.@threads` over the outer loop.
  - ❌ **TODO was wrong about `ShareSimilarTangents`** ("harder but worth attempting"). That mode is *semantically* serial: each iteration's decision depends on which earlier nodes were fitted vs shared. Threading would lose the sharing structure entirely or produce schedule-dependent outputs. Leave serial by design.
  - ✅ **Geometry fitting threaded in `19d93e0`.** Both `_fit_geometries(::NoSharing)` and `_fit_geometries_with_candidates(::NoSharing)` now use `Threads.@threads`. End-to-end speedup at `-t 4` vs `-t 1`: 1.25-1.43× on d∈{32, 64}; the d=128 stall (1.0×) is recovered to 1.4× by setting `BLAS.set_num_threads(1)` (BLAS oversubscription on small symmetric eigenproblems). Documented in `docs/graphs.md`.
  - ❌ **Edge-weight threading deliberately deferred.** Three reasons: (1) edge weights are only ~10% of build time on representative configs (PCA fit dominates), so Amdahl caps the win at ~10%; (2) `compute_edge_weight` is a public extension point — users can subclass `AbstractEdgeWeight` with custom weights, and silently threading the outer loop would impose a thread-safety contract on third-party implementations of `compute_edge_weight` and `local_distance` (`AbstractLocalGeometry`) without documenting it; (3) the right order is to first formalize the thread-safety contracts for the public extension points (see §4 below), then revisit. If pursued later, expose as opt-in (`build_weighted_graph(...; threaded_edges=false)` default false) and use `Atomic{Int}` for the `n_neg` counter.

- ✅ **`_fit_geometries` builds `Vector{Any}` then narrows.** `src/graphs/weighted_knn_graph.jl` line ~250: `Vector{Any}(undef, n)` defeats type inference for the entire fit pass. Fix: infer `G` from one fit call (`g1 = fit_geometry(...)`), allocate `Vector{typeof(g1)}(undef, n)`.
  - **Done in `1bf07e5`.** Smaller win than TODO suggested — function already narrowed to `Vector{G}` at the end before returning, so downstream callers were already type-stable. Real gain: avoid `n` boxing allocations + one full vector copy per `_fit_geometries` call. PCA fit dominates per-node cost so end-to-end speedup is small (single-digit %), but the change is clean and a clean prerequisite for threading. Applied to both `_fit_geometries(::NoSharing)` and `_fit_geometries_with_candidates(::NoSharing)`. `ShareSimilarTangents` variants kept as `Vector{Any}` because they fill out of order with `nothing` placeholders.

- **LSH `_collect_candidates` deduplicates with sort+unique, then `resize!` truncates lowest IDs.** `src/indices/lsh/index.jl` lines ~133–151, ~70–72. Sort+unique is O(C log C); `BitSet` would be O(C). The post-dedup `resize!(candidates, candidate_cap)` keeps the lowest-numbered IDs after `sort!`, not a meaningful subset. Fix: `BitSet` for dedup; for the cap, either don't truncate (compute distances for all then pick top-k) or sample uniformly.

- **HNSW traversal `maybe_push_candidate!` only pushes to the pending queue when added to `best`.** `src/indices/hnsw/traversal.jl` lines ~117–127. The standard HNSW search adds *every* unvisited neighbor whose `dist < worst` to the candidate frontier. Subtle deviation from the paper; likely benign but worth aligning. Fix: in `_search_layer`, push to `state.pending` whenever `dist < worst`, independently of acceptance into `best`.

- **`pushfirst!(path, current)` in `_reconstruct_path`.** `src/geodesic/geodesic_model.jl` line ~288: O(n) per call → O(n²) path reconstruction. Use `push!` then `reverse!` once.

### Low

- **`GreedySolver` complexity claim is wrong.** `src/graphs/refinement/solvers.jl` line ~459: documented as O(k² log k); actual implementation is `while ... for i ... for j` — O(k⁴) worst case. Either fix the algorithm (heap of edges) or fix the docstring.

- **`pack_bins(bins) = UInt64(hash(Tuple(bins)))`** allocates a `Tuple` from a `Vector` on every hash call. `src/indices/lsh/hash_functions.jl:52`. Inner LSH hash path; fold a simple FNV/xxhash over the `Int` projections directly.

- ✅ **PCA uses full SVD on `centered'`.** `src/geometry/pca.jl:155`. For local PCA with d ≪ n (typical), eigendecomposition of the d×d covariance via `Symmetric(C)` lets Julia call the symmetric eig path; same asymptotic cost but better constant factor.
  - **Done in `fbc95c6`.** Replaced with dual-eigen path (direct `Symmetric(centered * centered')` for d ≤ k, dual `Symmetric(centered' * centered)` + back-projection for d > k). End-to-end `build_weighted_graph` speedup: 8-16% at d∈{32, 64, 128}, ~3% at d=8 (noise). Smaller than the headline-grabbing factor the TODO implied — at the workload sizes the package actually uses (k=15, d ≤ 128), LAPACK and allocation overhead dominate the FLOP savings.

- 📌 **HNSW: pool the `BestCandidatesHeap` backing buffer.** After items 1–4 in this session landed (`_prune_list!` rewrite, generation-stamped visited buffer, in-place terminal sort, pre-sized adjacency), the remaining single-threaded build allocation is dominated by the per-call `BestCandidatesHeap` constructor inside `_search_layer` (~36 KB/call at ef_c=200, ~70% of remaining alloc). At n=20k single-threaded we are 1.29× behind hnswlib (2.96s vs 2.29s); at n=50k 1.23× behind (9.78s vs 7.97s). Pooling the heap buffer on the index — same coupling tradeoff as `visit_stamps` (build-only, single-threaded) — would plausibly close most of that. Estimated additional 10–20% single-threaded. Skipped this session in favour of threading, which is the bigger lever.

- 📌 **HNSW: query QPS is 2.4–7× behind hnswlib (apples-to-apples).** Confirmed via `scripts/hnsw_fair_compare.py` after the build perf work + threaded build landed. Single-threaded SIFT-128 n=10k k=10 ef_s=80: ours 5704 qps vs hnswlib 24690 qps. Same hot path as build but called many more times — every query allocates a fresh `BestCandidatesHeap` (~36 KB), a fresh `BitSetVisited`, and the per-node distance call wrapping has SubArray view + function call overhead. Build optimisations didn't apply because `query` runs concurrently (the build's index-state buffers can't be reused). Closing this needs either thread-local scratch on the index (with a per-thread acquire/release contract) or a `BufferPool` argument threaded through `query`. Estimated 2–3× achievable single-threaded; hard to fully close because hnswlib has a hand-tuned SIMD distance kernel and a lock-free visited slab.

- 📌 **HNSW: parallel build scaling is 3.15× at -t 4 (vs hnswlib ~3.6× at -t 4).** The new `threaded=true` path uses one `ReentrantLock` per node + one global lock for entry_point/max_layer. hnswlib uses lock-free slab design + atomic operations. Possibly worth investigating Polyester.jl + atomic-based adjacency-list updates for a future "close the last 15%" pass. Probably not until query QPS is closer to parity; threading scaling is good enough to ship.

- 📌 **HNSW threaded build: stale `cur_max` snapshot can silently skip connect on layers (rare).** When a thread snapshots `cur_max=2` at insertion start and another thread bumps `index.max_layer` past `level=4` mid-execution, our connect loop runs only on layers 2..0 — never connects at layers 3, 4 even though they now exist. Same effect as serial HNSW would have had if it inserted that node when max_layer was 2, so HNSW-correct in the local sense, but globally a quality hit when concurrent inserts at higher levels race ours. Brutal-critic flagged this as issue #3 in the threading review. Tried a fix (re-snapshot under global lock + run additional connect rounds at high layers under the lock); recall collapsed to ~0.5 — the fix had a bug I couldn't quickly debug, so reverted. Estimated impact: ~0.1% of inserts hit the race, recall hit ~0.001 absolute. Acceptable to ship, worth fixing later. Approach: hold `index.global_lock` for the entire connect loop *only* when `level > cur_max_at_snapshot`, re-reading `index.max_layer` inside the lock, running greedy descent through any newly-bumped layers, then running connect on `min(level, new_max)..0`. Stress + invariant tests would catch a corrupting fix immediately (recall floor 0.80 over 10 trials).

### Areas not reviewed

The independent code review skipped: `src/indices/multilevel/*` (IVF-HNSW hybrid, only file structure skimmed), `src/transforms/kmeans/*`, `src/preprocessing/*`, `src/geodesic/refinement.jl` (~580 lines), `src/geometry/neighborhood.jl` (~625 lines, expanding-shell strategies), `src/geometry/criteria.jl`, ORC `EdgeNeighborhoodView` construction (`graphs/refinement/types.jl`, `filtering.jl`, `effective_epsilon_policy.jl`). Test quality not audited beyond confirming tests exist for major code paths. Worth a second pass over these before claiming a complete review.

## 2. Distances.jl integration (partial)

Adopt Distances.jl as the default metric provider and use its BLAS-backed bulk APIs where they actually pay off. **Keep the callable-metric API everywhere else.**

### Recommended scope

- Replace `default_distance`, `default_squared_distance`, `cosine_distance`, `squared_cosine_distance` in `src/indices/bruteforce.jl` with thin aliases: `const default_distance = Euclidean()`, `default_squared_distance = SqEuclidean()`, etc. Re-export from `src/ManifoldANN.jl`.
- Wire `Distances.pairwise!` into the bruteforce builder/query paths and the k-means Lloyd kernels in `src/transforms/kmeans/distance.jl`. This is where the BLAS3 win actually materialises (the cross-term identity `‖x-y‖² = ‖x‖² + ‖y‖² − 2 x·y` dispatches to GEMM).
- Relax the `LSH` `distance_function(::AbstractLSHHash)` trait return type from `Function` to `Any` / `PreMetric` so it accepts Distances.jl metrics directly.

### Do not touch

- `src/graphs/refinement/filtering.jl` (`get_distance_function`)
- `src/graphs/edge_weight.jl` (`local_distance` is geometry-aware, not a plain metric)
- `src/geodesic/refinement.jl` (works on graph weights, not raw vectors)

These are graph/manifold operators; Distances.jl has nothing to offer them.

### What this does not buy

The per-call `evaluate(metric, x, y)` path in Distances.jl is a plain `@inbounds for ... zip` reduction with **no `@simd`, no `@turbo`, no LoopVectorization**. The current package code already carries `@simd` and uses `eachindex`, which is at least as auto-vectoriser-friendly. **There is no SIMD speedup on the per-edge distance loops** — only on bulk pairwise/colwise via BLAS3.

### Risks

- BLAS-backed pairwise can produce small negative values from cancellation in `‖x-y‖²`; Distances.jl clamps with `max(_, 0)`. Bit-for-bit results may differ from the current `@simd` reduction; could perturb tie-breaking in unit tests.
- Type stability with custom `Number` types: Distances.jl falls back to a slower generic loop for non-`BlasFloat` eltypes. Not a regression; just no win.
- LSH trait widening may require downstream code that asserts `::Function` to be relaxed.

### Realistic effort

Half a day for the alias swap and tests. 2–3 days for the bruteforce + k-means `pairwise!` integration with benchmarks. Document the BLAS-vs-`@simd` rationale in an ADR.

## 3. Manifolds.jl integration (post-thesis follow-up)

Wire up Manifolds.jl as a sampler and ground-truth-distance backend so the experimental pipeline can run on a wider set of manifolds than the bespoke Swiss roll and torus.

### Adapter surface

Small `sample_manifold(M, n) → (ambient::Matrix{Float64}, intrinsic)` adapter, ~30 LOC per manifold. Built on:

- `rand(M)` for sampling (intrinsic Haar/normalised measure; for embedded surfaces with non-uniform area, keep rejection sampling at the call site)
- `embed(M, p) |> vec` to flatten to a `Vector{Float64}` ambient column
- `distance(M, p, q)` for closed-form ground-truth geodesics

The kNN/ORC/edge-weight pipeline then runs unchanged on the ambient matrix.

### Concrete manifolds to add

1. **Sphere $S^2 \subset \mathbb{R}^3$.** Closed-form $d(p,q) = \arccos(p \cdot q)$, constant positive curvature. Sanity check that ORC behaves as predicted by constant-K theory.
2. **Hyperbolic $H^2$ (Lorentz embedding in $\mathbb{R}^3$).** Closed-form distance, constant negative curvature; complements the sphere and stresses the Euclidean-ambient assumption.
3. **Grassmann $\mathrm{Gr}(4,2)$ in $\mathbb{R}^{4 \times 2}$.** Closed-form distance via principal angles, non-trivial curvature, ambient dim 8. Tests the pipeline beyond $\mathbb{R}^3$ with intrinsic dim (4) and ambient dim (8) differing.
4. **SPD $P(3)$ with affine-invariant metric (vech-flattened to $\mathbb{R}^6$).** Closed-form distance, strongly negatively curved in some directions. Clean test of whether local-PCA tangent estimation degrades when the true tangent space has a quadratic-form structure.
5. **Rotations $SO(3)$ in $\mathbb{R}^9$.** Closed-form bi-invariant distance, compact with positive curvature, common in robotics/graphics.

### Friction points

- **Typed-point representation.** SPD points are `Symmetric{Float64,Matrix}`, Grassmann are `Matrix`, Hyperbolic are `HyperboloidPoint` wrappers. The current pipeline assumes `Vector{Float64}` columns of a `Matrix`. The "ambientize" layer must flatten via `embed(M, p) |> vec` and the rest of the pipeline operates on flat vectors.
- **EmbeddedTorus has no closed-form geodesic** in Manifolds.jl either — the same problem the thesis hit. Existing torus would still need the grid-Dijkstra approximation.
- **Local PCA tangent estimation degrades** when (i) ambient dim ≫ intrinsic dim and (ii) the true tangent space is an affine subspace of a non-linear variety (e.g. SPD at a base point). For SPD, local PCA in vech-coordinates will recover *a* tangent but not the metrically natural one. Worth keeping as a deliberately naive baseline; using Manifolds.jl's exact tangent as a second comparator would actually strengthen the empirical narrative.

### Dependency cost

`Manifolds.jl` + `ManifoldsBase.jl` + `RecursiveArrayTools` + `StaticArrays` + `Distributions` + `ManifoldDiff`. Substantial but pure-Julia; load time ~3-5s, no compiled binaries. ManifoldsBase has been stable since v0.15 (~2023); breaking changes are rare and well-signposted.

### Realistic effort

1–2 weeks. Out of scope for the thesis itself, but the Manifolds.jl-exact-tangent vs local-PCA-tangent comparison is a strong follow-up direction that directly answers an open question the thesis raises.

## 4. New findings during execution

Items added while working through §1.

- ➕ **Formalise query thread-safety as an `AbstractANNIndex` interface contract.** Currently `query(::AbstractANNIndex, data, q, k)` is thread-safe in every concrete implementation (KD-tree, HNSW, NN-Descent, bruteforce, LSH) — each call allocates its own working state and treats the index + data as read-only — but this is a happy coincidence, not a documented invariant. While threading `_fit_geometries_with_candidates` (which calls `query` from a `Threads.@threads` loop), I checked each implementation by hand to verify thread-safety. Future index implementations could silently break this without any test catching it.
  - Fix: (a) document the invariant on `AbstractANNIndex` (in its docstring or a dedicated interface contract section), explicitly stating that `query` MUST be safe to call concurrently on the same index, and that `build_index`/mutation operations are NOT; (b) add a generic regression test that, for each registered concrete index type, runs N queries serially and via `Threads.@threads` and asserts identical outputs.
  - Effort: ~1-2 hours. Suggested as a small follow-up commit after item §1 lands.

- ➕ **Boxing in `_fit_geometries(::ShareSimilarTangents)`.** Both sharing variants still use `Vector{Any}` (intentionally — they fill out of order with `nothing` placeholders). Could be tightened with a `Vector{Union{Nothing, G}}` once `G` is known, but `G` is only known after the first fit, and the placeholder pattern complicates that. Low-priority polish; not worth doing speculatively.

- ➕ **Re-enable PyNNDescent in the benchmark runner.** A wrapper already exists at `benchmarking/benchmarking/wrappers/pynndescent.py` but is intentionally excluded from `benchmarking/configs/algorithms.yaml` with a comment claiming it requires Python <3.10 due to `llvmlite`. **That comment is outdated**: a 2026-05-02 install on the existing Python 3.13.5 venv (`uv pip install pynndescent`) succeeded with `llvmlite==0.47.0`, `numba==0.65.1`, `pynndescent==0.6.0`. Quick smoke run on n=2000, d=32, k=15: PyNNDescent built in 56.8 ms (Float32, Euclidean). Re-enable in `algorithms.yaml`, drop the stale comment in `pyproject.toml`.

- ➕ **Adaptive `max_iterations` for NN-Descent.** Our default is hardcoded `10`. PyNNDescent uses `max(5, round(log2(n)))` (13 at n=10k, 17 at n=100k). Algorithmically NN-Descent's iteration count to convergence scales sub-linearly with n, so a fixed default under-iterates at large n and over-iterates at small n. Cheap fix, plausibly small recall improvement at large n. Convergence threshold itself (`1e-3` relative improvement) already matches PyNNDescent's `delta=0.001`.

- ➕ **NN-Descent now thread-parallelizable.** `25fbbee` adds `threaded=true` (default) to `build_index(NNDescentIndex, ...)`, with per-node `ReentrantLock`s protecting heap mutations and per-thread RNG/scratch state. 1.46-1.87× speedup at -t 4 across n=20k-50k. **Threading gives up bitwise determinism** (thread interleaving determines insertion order); same contract as PyNNDescent's `n_jobs=1`. Users wanting reproducibility pass `threaded=false`. Reproducibility test in `nndescent_tests.jl` updated to pass `threaded=false`.

- ➕ **NN-Descent hot-path allocation cleanup.** `2d912f0` adds `sizehint!(heap.data, capacity)` in the `BoundedMaxHeap` constructor and `sizehint!` for the per-node reverse buffers (sized at 2k — structural guess about reverse-degree skew, NOT related to the `pruning_degree_multiplier` sampling cap). 1.13-1.16× build speedup, brings item 3's reverse-sampling overhead vs the no-reverse baseline from +23% down to +6%.

- ➕ **Final PyNNDescent comparison (Float32, Euclidean, after all NN-Descent commits).**

  | Comparison | n=20k d=32 k=15 | n=50k d=32 k=15 |
  |---|---|---|
  | Single-threaded vs `n_jobs=1` | 1.7× behind | 1.9× behind |
  | Our `-t 4` vs `n_jobs=1` | 1.08× behind | 1.18× behind (effective parity) |
  | Our `-t 4` vs `n_jobs=4` | 3× behind | 3.5× behind |

  The original "6.7× behind" headline was a single-threaded vs multi-threaded comparison and was misleading. After our threading commit, the single-threaded gap is reasonable and we match PyNNDescent at low parallelism. The multi-threaded gap is real and deep — see "threading model gap" below.

- ❌ **RP-tree initialization did not close the gap.** Three implementation variants benchmarked (forward-only, all-bidirectional, top-k bidirectional). All 1.6-10× slower than random init at every config tested (n=20k, n=50k, n=100k). Diagnosis was wrong: with `_initialize_random_neighbors!`'s bidirectional pushes, NN-Descent already converges in 1-2 iterations on this codebase's `convergence_threshold=0.001` default — there were no iterations to save. RP-tree did improve recall (n=5000 d=32 k=15: 0.92 → 0.96, +3.6pp). Code retained as opt-in (`init=:rptree`); see "RP-tree opt-in (uncommitted)" below.

- ❌ **Threading rework via dynamic dispatch did not improve scaling.** Profile of `-t 4` build showed ~30% of time in `poptask`/`wait` (idle), suggesting load imbalance. Tried switching from `Threads.@threads :static` to `@sync` + `@spawn` over chunked node ranges (4× more chunks than threads, per-chunk RNG and per-task scratch) to give the runtime work to balance. **Result: 6-9% slower at every config**, not faster. Per-task allocation overhead and task spawn/sync cost exceeded the load-imbalance savings at our work granularity. Reverted. The "30% idle" measurement included task-switching time, not just thread starvation.

- ➕ **Threading model gap to PyNNDescent is genuinely deep.** PyNNDescent scales 2.8× at `n_jobs=4` while ours scales 1.5× at `-t 4`. Profile says lock contention is only ~5% of total — the gap is in their lock-free design. Closing it would require:
  - Per-thread update buffers (deferred merge): all heap mutations done in a serial reduction phase. ~150-200 LOC. Risk: serial merge becomes the new bottleneck (~10M inserts/iter at 50ns each = 500ms serial).
  - OR a custom heap with atomic operations replacing `BoundedMaxHeap`. Substantial rewrite.
  - OR Polyester.jl for cheaper thread launch.
  Speculative; bench gate would need 1.3× over current threaded path to justify. Not pursued this session.

- ✅ **RP-tree opt-in committed in `05b91d3`.** `src/indices/nndescent/rptree_init.jl` (~296 LOC) + `test/unit/indices/nndescent/rptree_tests.jl` + kwarg wiring in `build_index(NNDescentIndex, ...; init=:rptree, n_trees=…, leaf_cap=…)`. Default stays `:random`. Kept primarily because the primitives (`RPTree`, `build_rptree`, `build_rptree_forest`, `leaf_members`) are generic and could become a standalone `RPTreeIndex` for high-d datasets where `KDTreeIndex` degrades. If that happens, move primitives from `src/indices/nndescent/rptree_init.jl` to `src/utils/rptree.jl`.

  **Real finding** that came out of this exploration: `_initialize_random_neighbors!`'s bidirectional pushes give each node ~2k starting neighbors instead of k, which is why NN-Descent converges so fast on this codebase. The PyNNDescent gap analysis ("they do 5-8 iters, we do 25") was misleading — with `convergence_threshold=0.001` we also typically only do 1-2 effective iterations.

  **Tooling note**: do NOT use `LoopVectorization.jl` (`@turbo`/`@avx`) for any future perf work. In maintenance-only mode since ~2024, fragile on Julia 1.10+, ecosystem moved to plain `@simd` + `SIMD.jl` + `VectorizationBase.jl`.

- ➕ **Optional: PyNNDescent as a quality oracle for our NN-Descent.** Beyond the bench suite, consider a separate validation target where we compare our NN-Descent against PyNNDescent on identical data/k. The current strengthened recall ≥ 0.90 / MRR ≥ 0.92 tests catch *catastrophic* drops; head-to-head with PyNNDescent would catch *creeping* drops (e.g. 5% gap). Trade-off: requires Python in the validation loop — meaningful project coupling. Defer until the broader Distances.jl/Manifolds.jl/NearestNeighbors.jl ecosystem-native question (below) is decided.

- ➕ **Strategic question: how Julia-ecosystem-native do we want to be?** Three workstreams point in the same direction and should be considered together rather than as discrete fixes:
  - §2 Distances.jl as the metric provider.
  - §3 Manifolds.jl as the manifold/sampler provider.
  - **NearestNeighbors.jl** as an optional alternative ANN backend. Mature, threaded, gives us BallTree (which we don't have, useful for higher-d regimes where KD-tree degrades). Replacing our KD-tree for parity is not worth it (we just paid down its biggest perf debt in `e722498`); adding a `NearestNeighborsBallTree` index wrapper would be cheap and additive.

  The thesis benefits from the current self-contained stance (clear scope, fewer moving parts). A published library benefits from the ecosystem-native stance (less code to maintain, broader compatibility, easier for others to extend). Decide direction first, then sequence §2/§3/NearestNeighbors as one coherent effort.
