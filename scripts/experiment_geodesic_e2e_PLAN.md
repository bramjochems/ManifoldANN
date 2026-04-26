# Plan: `experiment_geodesic_e2e.jl`

End-to-end pipeline evaluation of the curvature-free geodesic estimator from
Chapter 6 of the thesis. This document specifies the design of a future Julia
script `scripts/experiment_geodesic_e2e.jl` that produces the empirical
content of §6.4 (`\section{End-to-end pipeline evaluation}`,
`\label{sec:geod-discussion}`) of
`docs/thesis/content/chapter-geodesic.tex`.

The script is a **thesis experiment**, not a package component. It lives in
`scripts/`, has no unit tests, and is permitted to be a self-contained
linear procedural script. It must, however, be reproducible: deterministic
seeds, pinned dataset parameters, timestamped output directory, a
`config.toml` recording the run.

**Package dependency.** The script consumes a first-class corrected-geodesic
estimator API that is being added to the package in parallel: an
`AbstractEdgeGeodesicEstimator` trait in `src/geodesic/`, with concrete
subtypes `EuclideanChord`, `TangentProjectedSymmetricMean`, and
`CurvatureFreeSymmetric` (the latter implements the symmetrised
$\hat d_g^{\mathrm{sym}} = (8 d_E - d_T^{(x)} - d_T^{(y)})/6$ from §6.2.3
of the thesis, eq. `eq:geod-sym`). These plug into
`build_geodesic_model(...; edge_estimator=...)`. If the API has not landed
by the time this experiment is implemented, the implementer should stub it
out, but the script itself commits to the clean shape — no inline weight
arithmetic, no hand-rolled non-negativity clamps. Negative or otherwise
ill-defined edge weights from the corrected estimator are the package's
problem; the script only logs the count returned by the estimator.

---

## 1. Goal and scope

The verification in §6.2 establishes the per-edge claim: a single edge
weighted by $\hat d_g^{\mathrm{sym}}$ has $O(\ell^4)$ error in the geodesic
arc length $\ell$, an order improvement over the $O(\ell^3)$ error of either
$d_E$ or $d_T$. The §6.4 question is whether this per-edge gain transmits
through shortest-path aggregation.

In §6.4 the chapter names two error sources for the end-to-end MRE:

  1. **Edge-weight bias** — the per-edge estimator's deviation from the
     true geodesic arc length along that one edge.
  2. **Path-deviation bias** — the off-geodesic detour incurred because
     the shortest path through samples is not the smooth manifold
     geodesic. This is positive and (to leading order) estimator-
     independent at fixed graph topology and fixed *path*.

This experiment recognises a third source that the chapter's two-source
decomposition glosses over:

  3. **Path-selection bias** — changing the edge weights changes which
     vertex sequence Dijkstra picks. Even at fixed graph topology, two
     weight schemes can traverse different edges, so the path-deviation
     contribution is not actually estimator-independent in practice.

The experiment is designed to attribute the observed MRE to all three. To
do that cleanly we run **four** weight schemes on the *same* kNN graph
topology, including a fourth analytic-distance baseline that uses the
true $d_{\mathcal{M}}$ as the edge weight. The baseline by construction
has zero edge-weight bias and picks the path that minimises true
arc-length sum; we treat it as the "ideal path" for the given graph.

For each `(n, scheme, pair)` we record three lengths so the three error
sources can be teased apart:

  - `estimated_distance` — what Dijkstra returns under the chosen scheme.
  - `chosen_path_analytic_length` — sum of $d_{\mathcal{M}}$ across the
    edges of the path Dijkstra actually picked under this scheme.
  - `ideal_path_analytic_length` — `estimated_distance` of the analytic
    baseline scheme for the same `(n, rep, pair)`.

The decomposition is then:

  - **Edge-weight error** = `estimated_distance` − `chosen_path_analytic_length`.
    What the per-edge bias contributes once the path is fixed.
  - **Path-selection error** = `chosen_path_analytic_length` −
    `ideal_path_analytic_length`. What the scheme costs us by routing
    through suboptimal vertices.
  - **Path-deviation error** = `ideal_path_analytic_length` − `true_geodesic`.
    The unavoidable off-geodesic detour at this density. By construction
    this column is the same for all four schemes within a single
    `(n, rep, pair)` (it is the analytic-baseline residual).

A successful experiment produces clean slope separation in the headline
MRE plot and a clear attribution of the residuals in the decomposition
plot. A null result (all four slopes within noise of each other) is
itself a valid finding for §6.4.

---

## 2. Data generation

**Manifold.** Swiss roll, the parametric form already used in §5.1 of
`chapter-experiments.tex`:

```
x(t, v) = [ t cos t,  v,  t sin t ]
t ∈ [1.5π, 4.5π],  v ∈ [0, 10]
```

**Reuse the existing implementation.**
`generate_swiss_roll(n; rng, t_min, t_range, h_scale)` from
`docs/examples/geodesic/swiss_roll_utils.jl` already does exactly this with
the right defaults. Include via:

```julia
include(joinpath(@__DIR__, "..", "docs", "examples", "geodesic",
                 "swiss_roll_utils.jl"))
```

(matching the convention in `experiment_orc.jl`).

**Sample sizes.** $n \in \{200, 350, 500, 800, 1200, 2000, 4000, 8000\}$.
The lower end is added because that is where the predicted $-2$ vs $-3$
slope gap is most visible; in the previous 5-value grid the small-$n$
regime was undersampled for slope estimation.

**Sampling.** Uniform in parameter space $(t, v)$. No noise. §6.4 frames
this as a clean-data study of edge-weight aggregation; tangent-plane
noise is §6.5 future work.

**Seeding.** Two independent seed namespaces, deliberately disjoint:

  - **Point-cloud RNG.** `data_seed(n, r) = BASE_SEED_DATA + 1_000_000*r + n`
    with `BASE_SEED_DATA = 0xC0DE_6E0`.
  - **Pair-selection RNG.** `pair_seed(n, r) = BASE_SEED_PAIRS + 1_000_000*r + n`
    with `BASE_SEED_PAIRS = 0xPA1R_6E0` (literal: `0x9A1C_6E0`).

Two separate `BASE_SEED_*` constants and a multiplier of $10^6$ on the
rep index guarantee no collisions across the planned $(n, r)$ grid (the
largest $n$ is 8000, so adding $n$ never crosses the next rep block).
The script asserts this at startup.

---

## 3. Source-target pair selection

§6.4 wants pairs whose true geodesic is "non-trivial". Concrete protocol:

1. **Reference distribution, computed once at the start of the run.**
   At a separate large reference draw $n_{\mathrm{ref}} = 4000$ with
   `MersenneTwister(BASE_SEED_DATA - 1)`, sample
   $K_{\mathrm{ref}} = 50{,}000$ random index pairs, compute their
   analytic geodesic distances via `exact_swiss_roll_geodesic`, and
   take the empirical Q25 and Q75 quantiles. These quantiles
   `[d_lo, d_hi]` define the **acceptance band**.
2. **Per-`(n, rep)` selection.** Pairs are regenerated per `(n, rep)`
   because pair indices index into a point set that changes with
   $n$. Using the per-`(n, rep)` pair RNG: draw candidate pairs
   uniformly without replacement from $\binom{n}{2}$, compute their
   analytic geodesic, keep those falling in `[d_lo, d_hi]`, stop when
   `N_PAIRS = 100` accepted pairs have been collected. To bound
   worst-case effort, cap candidate draws at `10 * N_PAIRS`; if fewer
   than `N_PAIRS` are accepted (vanishingly unlikely for the middle
   50% band), warn and proceed with whatever was accepted.
3. **Number of pairs.** $N_{\mathrm{pairs}} = 100$.
4. **Distinctness.** Pairs are stored as `(src, tgt, d_true)` tuples
   with `src < tgt`.

Using the middle 50% band exercises typical pair distances rather than
only long ones (the previous "upper third" filter biased towards the
worst-case path-deviation regime).

---

## 4. Pipeline

For each `(n, rep)`:

1. **Generate data.** `data, params = generate_swiss_roll(n;
   rng=MersenneTwister(data_seed(n, rep)), t_min=1.5π, t_range=3π,
   h_scale=10.0)`.
2. **Build ANN index and kNN graph at $k = 15$.**
   - `index = build_index(BruteForceIndex, data)` (brute force is fine at
     $n \le 8000$).
   - `graph = build_knn_graph(index, data; k=15, directed=false)`.
3. **Symmetrisation convention.** `directed=false` is documented in
   `experiment_orc.jl` to use the **union** of $i \to j$ and $j \to i$
   neighbour edges (an edge exists if *either* direction is in the
   $k$-NN list). The implementer must verify this against the package
   source and pin the convention in the `config.toml` as
   `symmetrisation = "union"`. The same convention is used for ORC, so
   end-to-end MRE numbers are directly comparable across experiments.
4. **Build a `GeodesicModel` per scheme using the package API.** See §4a.
5. **Run path-recovering Dijkstra from every source to every target.**
   See §4b.
6. **Record results.** One CSV row per
   `(n, rep, scheme, pair_id)`.

### 4a. Weight schemes (four)

The first three are constructed via the package API. They share one
underlying `graph` (same kNN topology) and differ only in
`edge_estimator`:

1. **`d_E`** — Euclidean chord.
   `model = build_geodesic_model(graph, data; edge_estimator=EuclideanChord())`.
2. **`d_T_sym`** — symmetric tangent-projected.
   `model = build_geodesic_model(graph, data;
       edge_estimator=TangentProjectedSymmetricMean())`.
3. **`d_g_hat_sym`** — corrected symmetric estimator
   $\hat d_g^{\mathrm{sym}} = (8 d_E - d_T^{(x)} - d_T^{(y)})/6$
   from §6.2.3.
   `model = build_geodesic_model(graph, data;
       edge_estimator=CurvatureFreeSymmetric())`.

   No inline arithmetic. No non-negativity clamp. The package owns the
   policy for what to do with negative or otherwise ill-defined edge
   weights (skip / floor / error). The script reads back from the
   constructed model the count of edges flagged as such (`model.n_neg_edges`
   or equivalent — exact field name to be confirmed against the parallel
   API change) and logs it per `(n, rep)` to the run log; the count is
   also written to `summary.csv` per `(n, scheme)` aggregated.

4. **`d_analytic`** — analytic-distance baseline. **Not** routed through
   `build_geodesic_model`. Compute the weight dict directly:

   ```julia
   weights = Dict{Tuple{Int,Int}, Float64}()
   for (i, nbrs) in enumerate(graph)
       for j in nbrs
           if i < j
               weights[(i, j)] = exact_swiss_roll_geodesic(params, i, j)
           end
       end
   end
   ```

   and pair this with the same adjacency structure as the other three
   schemes (e.g. via `graph_to_adj_weights`-style construction but with
   weights replaced). This is the *path-deviation-only* scheme: by
   construction the edge weight is the true arc length, so the only
   error left is the off-geodesic detour through samples.

The intrinsic dimension passed to whatever local-PCA fitting
`build_geodesic_model` does internally is 2. If the API does not auto-fit
PCA, the script must fit `PCAGeometry` per node and pass the geometries in
explicitly; see §5.

### 4b. Dijkstra with path recovery

Every recorded distance must come with the **vertex sequence** of the
shortest path Dijkstra picked, because we need it to compute
`chosen_path_analytic_length` (sum of analytic geodesic arc lengths over
the path's edges).

`scripts/orc_helpers.jl::dijkstra_from(adj, weights, source, n)` may or
may not currently return predecessor information. The implementer must:

  - Inspect `dijkstra_from`. If it already returns a predecessor vector
    or path, use it.
  - Otherwise, write a small path-recovering Dijkstra inline in this
    script (binary heap, predecessor array, reconstruction function).
    Do not modify `orc_helpers.jl` for a one-off thesis experiment;
    duplicating ~30 lines is cleaner than coupling.

For each `(n, rep, scheme)`: run single-source Dijkstra from each
distinct source in the pair list (≤ 100 unique sources per rep), keep the
predecessor array, and for each `(src, tgt)` pair reconstruct the path
and compute its analytic length.

---

## 5. Tangent-plane estimation

If the new `build_geodesic_model` API encapsulates PCA fitting, pass
`intrinsic_dim=2` and let it. If not, fit explicitly:

```julia
pca_method = PCAMethod(intrinsic_dim=2)
geometries = Vector{PCAGeometry}(undef, n)
for i in 1:n
    nbrs = graph[i]
    geometries[i] = fit_geometry(pca_method, data, i, nbrs)
end
```

(matching `experiment_orc.jl` lines 307-319), and pass `geometries` into
`build_geodesic_model`.

PCA neighbourhood = graph kNN neighbourhood ($k = 15$). Decoupling the
two would introduce a hyperparameter §6.4 does not vary.

---

## 6. Ground truth

Closed-form Swiss roll geodesic from §5.1 of
`chapter-experiments.tex`: with arc-length function
$s(t) = \tfrac12 (t\sqrt{1+t^2} + \sinh^{-1}(t))$,
$$
d_{\mathcal{M}}((t_1, v_1), (t_2, v_2)) = \sqrt{(s(t_2) - s(t_1))^2 + (v_2 - v_1)^2}.
$$
Reuse `exact_swiss_roll_geodesic(params, i, j)` from
`docs/examples/geodesic/swiss_roll_utils.jl`. Used both for ground truth
and for the analytic-baseline edge weights and for the
`chosen_path_analytic_length` along reconstructed paths.

---

## 7. Metrics

For each `(n, rep, scheme, pair)` the script records four lengths:

  - `estimated_distance`
  - `chosen_path_analytic_length`
  - `ideal_path_analytic_length` (`estimated_distance` of the
    `d_analytic` scheme for the same `(n, rep, pair)`)
  - `true_geodesic`

and four (signed or absolute, see below) error columns:

  - `relative_error` — `|estimated − true_geodesic| / true_geodesic`.
    The headline MRE.
  - `edge_weight_error` —
    `(estimated − chosen_path_analytic_length) / true_geodesic`.
    Signed; the corrected estimator can over- or under-shoot.
  - `path_selection_error` —
    `(chosen_path_analytic_length − ideal_path_analytic_length) /
    true_geodesic`. Always ≥ 0 by definition of "ideal".
  - `path_deviation_error` —
    `(ideal_path_analytic_length − true_geodesic) / true_geodesic`.
    Always ≥ 0 (sampled paths are at least as long as the smooth
    geodesic). Same value across all four schemes for a given
    `(n, rep, pair)`; recorded redundantly for clarity.

The signed components sum to `(estimated − true_geodesic) / true_geodesic`,
i.e. the signed version of the headline error.

**Outlier handling.**

  - **Disconnected pairs.** If Dijkstra returns `Inf` for a target the
    pair is excluded from the MRE for that `(n, rep, scheme)` and
    counted in `n_disconnected`. The CSV row is still written with
    `disconnected = 1` and `NaN` errors.
  - **Negative-weight edges.** Handled by the package; just logged.
  - **`d_analytic` failure.** If the analytic baseline disconnects a
    pair (geometric impossibility on this graph topology — a kNN graph
    is the same regardless of weights), the pair is excluded for *all*
    schemes for that `(n, rep)`, because the decomposition is
    ill-defined without `ideal_path_analytic_length`.

---

## 8. Repetitions

`N_REPS = 10`. Doubling the previous 5 reps gives a more usable cluster
bootstrap; runtime is still well within budget (see §11).

---

## 9. Output

```
benchmark_results/geodesic_e2e/{timestamp}/
    results.csv      # raw rows (one per pair × rep × scheme × n)
    summary.csv      # one row per (n, scheme)
    config.toml      # reproducibility metadata
    mre_vs_n.{png,pdf}              # produced by the plotter
    decomposition_vs_n.{png,pdf}    # produced by the plotter
```

`timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")`. Base directory
`benchmark_results/geodesic_e2e/` is created with `mkpath`.

### 9a. `results.csv` columns

| column                          | type    | description |
|---------------------------------|---------|-------------|
| `n`                             | Int     | sample size |
| `rep`                           | Int     | repetition index, 0..N_REPS-1 |
| `data_seed`                     | Int     | point-cloud seed |
| `pair_seed`                     | Int     | pair-selection seed |
| `scheme`                        | String  | `d_E`, `d_T_sym`, `d_g_hat_sym`, `d_analytic` |
| `pair_id`                       | Int     | 0..N_PAIRS-1 within `(n, rep)` |
| `src`                           | Int     | source node |
| `tgt`                           | Int     | target node |
| `path_length`                   | Int     | number of edges in the chosen path; `0` if disconnected |
| `estimated_distance`            | Float64 | Dijkstra distance under this scheme; `NaN` if disconnected |
| `chosen_path_analytic_length`   | Float64 | sum of $d_{\mathcal{M}}$ over chosen-path edges |
| `ideal_path_analytic_length`    | Float64 | `estimated_distance` of `d_analytic` for this pair |
| `true_geodesic`                 | Float64 | analytic Swiss-roll $d_{\mathcal{M}}(src, tgt)$ |
| `relative_error`                | Float64 | headline MRE term |
| `edge_weight_error`             | Float64 | signed edge-weight component, see §7 |
| `path_selection_error`          | Float64 | path-selection component, see §7 |
| `path_deviation_error`          | Float64 | path-deviation component, see §7 |
| `disconnected`                  | Int     | 0 or 1 |

Numeric formatting: `@sprintf("%.6e", v)` for floats.

### 9b. `summary.csv` columns

One row per `(n, scheme)`:

| column                      | type    | description |
|-----------------------------|---------|-------------|
| `n`                         | Int     | |
| `scheme`                    | String  | |
| `n_pairs_used`              | Int     | total non-disconnected pairs across reps |
| `n_disconnected`            | Int     | total disconnected pairs across reps |
| `n_neg_edges_total`         | Int     | total edges flagged by the estimator |
| `mre_mean`                  | Float64 | mean over all valid pairs |
| `mre_ci_lo`                 | Float64 | cluster bootstrap 2.5th percentile |
| `mre_ci_hi`                 | Float64 | cluster bootstrap 97.5th percentile |
| `edge_weight_mre_mean`      | Float64 | mean of `|edge_weight_error|` |
| `path_selection_mre_mean`   | Float64 | mean of `path_selection_error` |
| `path_deviation_mre_mean`   | Float64 | mean of `path_deviation_error` (same across schemes per `n`) |

**Cluster bootstrap.** Resample *reps* (not pairs) with replacement
$B = 1000$ times. For each resample, recompute the per-pair-pooled MRE
across the resampled reps and take the 2.5/97.5 percentiles. This is
the right level for the inferential question (one rep is one independent
draw of the experiment).

### 9c. `config.toml`

Use the existing `write_config_toml` helper from `orc_helpers.jl`, extended
to include:

  - `base_seed_data`, `base_seed_pairs`, `n_grid`, `n_reps`, `k`,
    `n_pairs`, `t_min`, `t_range`, `h_scale`, `intrinsic_dim`,
    `schemes`, `bootstrap_b`, `symmetrisation`
  - `julia_version` from `VERSION`
  - `pkg_status` — output of `Pkg.status(; io=IOBuffer())` captured to
    a string, recording exact pinned versions of all direct deps
  - `git_hash` (already auto-included)

If `write_config_toml` doesn't yet record `julia_version` and
`pkg_status`, extend it (these are useful for the ORC experiment too).

---

## 10. Plotting

A separate Python script `scripts/plot_geodesic_e2e.py`. Input: a
`benchmark_results/geodesic_e2e/{timestamp}/` directory (CLI argument;
default to most-recent). Output: two figures, each as PNG (300 dpi) and
PDF.

### Figure 1 — `mre_vs_n.{png,pdf}` (headline)

Log-log plot of MRE vs $n$:

  - x-axis: $n$, log scale.
  - y-axis: `mre_mean`, log scale.
  - Four lines, one per scheme: `d_E`, `d_T_sym`, `d_g_hat_sym`,
    `d_analytic`. Markers + lines, distinct colours; the `d_analytic`
    line is dashed (it is a baseline, not a contender).
  - Shaded bands from `mre_ci_lo`, `mre_ci_hi`.
  - Reference slope guides anchored at the smallest-$n$ point of the
    `d_T_sym` curve: dashed line of slope $-2$ (per-edge bias of
    $d_T$/$d_E$) and dotted line of slope $-3$ (per-edge bias of the
    corrected estimator). Labelled in the legend.
  - Title: "End-to-end MRE vs sample size, Swiss roll, $k = 15$".

For each scheme, fit $\log(\mathrm{MRE})$ vs $\log(n)$ via
`numpy.polyfit`, report fitted slope + bootstrap SE in the legend label,
and dump to `slopes.csv`.

### Figure 2 — `decomposition_vs_n.{png,pdf}` (diagnostic)

A 1×3 panel of log-log plots (or three subplots stacked vertically),
one per error component:

  - Panel A: `edge_weight_mre_mean` vs $n$, four schemes (the
    `d_analytic` curve is at machine zero by construction; include it as
    a sanity check).
  - Panel B: `path_selection_mre_mean` vs $n$, four schemes (the
    `d_analytic` line is identically zero; include for reference).
  - Panel C: `path_deviation_mre_mean` vs $n$. This is one curve (same
    value across schemes per $n$); plot once.

Same colour key as Figure 1.

Use the same plotting style conventions as `analyze_orc.py`.

---

## 11. Runtime expectations

Dominant cost is path-recovering Dijkstra: per `(n, rep, scheme)`, ≤ 100
single-source Dijkstras on a graph with $|V| = n$, $|E| \approx 15n$.

At $n = 8000$: ~$10^5$ heap ops/source × 100 sources × 4 schemes × 10
reps ≈ $4 \times 10^8$ ops, roughly 1-2 minutes in compiled Julia with
predecessor tracking.

Other costs at $n = 8000$: brute-force kNN (~$2 \times 10^8$ ops, a few
seconds); local PCA at every node (well under a minute). Pair selection
and analytic geodesics are negligible.

**Total wall-clock end to end:** roughly 15-30 minutes on a modern
desktop. A simple linear loop is fine; no parallelisation needed beyond
BLAS threading.

---

## 12. What this script does NOT do

  - **Other manifolds.** Swiss roll only.
  - **Noise scans.** $\sigma = 0$ throughout (§6.5 future work).
  - **Asymmetric estimator $\hat d_g^{(x)}$.** Only the symmetrised
    form, matching §6.4.
  - **ANN benchmarking.** A single brute-force index is used.
  - **Sweep over $k$.** Only $k = 15$.
  - **Pruning.** None.
  - **Other weight schemes.** Only the four named above.

---

## 13. CLI / environment variables

Mirror `experiment_orc.jl`'s style. Only the env vars that are actually
needed:

  - `N_REPS_OVERRIDE=2` — override the rep count (for quick checks).
  - `OUTPUT_DIR=path` — override the timestamped output directory
    (useful for tests and for the plotter pointing at a known location).

No `SMOKE`, no `N_OVERRIDE`, no resume logic. The full run is short
enough that overrides for individual quick checks are sufficient.

---

## 14. Predictions

The chapter's slope predictions ($-2$ for $d_E$/$d_T$, asymptoting
toward $-3$ for $\hat d_g^{\mathrm{sym}}$, all curves bottoming out at a
shared path-deviation floor at large $n$) describe the **expected shape
under the assumption that path-selection effects are negligible**. The
experiment measures this assumption directly via Figure 2.

  - **Edge-weight component (Panel A).** This is the right column to
    test the chapter's slope claims: it isolates the per-edge bias by
    fixing the path. Expected slopes: $\approx -2$ for `d_E` and
    `d_T_sym`, $\approx -3$ for `d_g_hat_sym`, machine-zero floor for
    `d_analytic`.
  - **Path-deviation component (Panel C).** A single curve; expected to
    decay slowly (the chapter conjectures $O(n^{-2/d})$ for intrinsic
    dimension $d = 2$, but does not pin this down). This is the floor
    that all four schemes asymptote to in Figure 1 once edge-weight
    bias has decayed away.
  - **Path-selection component (Panel B).** *No prediction.* This is
    the diagnostic that tells us whether the chapter's two-source
    framing was adequate. Three plausible outcomes:
      a. Path-selection is negligible across all schemes — the
         chapter's predictions hold and the headline plot looks like
         the chapter sketch.
      b. Path-selection is non-negligible and similar across schemes
         (e.g. all positive but small) — the chapter's predictions
         hold qualitatively but the headline curves are uniformly
         shifted up.
      c. Path-selection is differential across schemes (the corrected
         estimator picks materially different paths) — the chapter's
         slope claims may be partially obscured in Figure 1, but
         Panel A still shows the per-edge gain.

A null result on Panel A (no slope separation) would falsify the
per-edge claim's transmissibility through aggregation. A successful
result on Panel A but a swamped result on Figure 1 would identify
path-selection as the explanation and motivate a follow-up on
geodesic-aware path selection.

---

## 15. Cross-references

  - **§6.4 of the thesis** — `docs/thesis/content/chapter-geodesic.tex`,
    `\section{End-to-end pipeline evaluation}`.
  - **§6.2.3** (corrected estimator definition, eq. `eq:geod-sym`) —
    same file.
  - **Swiss roll setup and ground-truth formula** —
    `chapter-experiments.tex` §5.1.
  - **Style reference** — `scripts/experiment_orc.jl` for output
    layout, env-var conventions, console banner, CSV-flush pattern.
  - **Helpers to reuse** — `scripts/orc_helpers.jl` for `init_csv`,
    `append_csv_rows`, `write_config_toml`, `graph_to_adj_weights`,
    and (if it returns predecessors) `dijkstra_from`. *Do not
    duplicate these.* If `dijkstra_from` does not return paths, write
    a small path-recovering Dijkstra inline in this script — do not
    edit `orc_helpers.jl` for a one-off experiment.
  - **Swiss roll generator and analytic geodesic** —
    `docs/examples/geodesic/swiss_roll_utils.jl`. *Do not duplicate.*
  - **New geodesic estimator API** — `src/geodesic/` (parallel package
    change). Subtypes `EuclideanChord`,
    `TangentProjectedSymmetricMean`, `CurvatureFreeSymmetric` of
    `AbstractEdgeGeodesicEstimator`, consumed by
    `build_geodesic_model(...; edge_estimator=...)`.

---

## 16. Done when

  - Running `julia --project=. -t auto scripts/experiment_geodesic_e2e.jl`
    produces a fresh timestamped directory with `results.csv`,
    `summary.csv`, `config.toml`.
  - `summary.csv` has one row per `(n, scheme)` for the eight $n$
    values and four schemes (32 rows in the nominal case).
  - The Python plotter on that directory produces both
    `mre_vs_n.{png,pdf}` and `decomposition_vs_n.{png,pdf}` with all
    four schemes visible and the slope guides drawn.
  - `config.toml` records seeds, Julia version, `Pkg.status()` output,
    git hash, and the symmetrisation convention.

The thesis discussion in §6.4 is then written from `summary.csv`,
`slopes.csv`, and the two figures.
