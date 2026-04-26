# ORC-ManL validation against the reference Python `orcml` package

This document describes how to reproduce the cross-implementation
validation reported in §4.2 of the thesis (the scatter / Bland-Altman
figure and the bias diagnosis).

## What is being validated

The Julia ORC-ManL implementation in `ManifoldANN.jl` against the
reference Python implementation in `benchmarking/external/orcml/`
(which delegates the actual OT computation to the
`GraphRicciCurvature` package, called with `weight="effective_eps",
alpha=0.0, method='OTD'`).

The headline number reported in the thesis is Pearson `r = 0.9981`
across 4326 edges of a 500-point swiss roll (`k = 15`,
noise `σ = 0.05`), with a small systematic bias of `−0.037`.

## Reproducing the agreement plots

These steps assume the Python virtualenv at `benchmarking/.venv` has
already been created (see `benchmarking/setup.sh` if not) and Julia is
installed.

1. Generate the dataset and the reference curvatures from the orcml
   package (writes to `benchmark_results/`):

   ```bash
   cd benchmarking
   .venv/bin/python ../scripts/generate_orcml_validation_data.py \
       --n-points 500 --noise 0.05 --k 15 --seed 42
   ```

2. Compute the corresponding curvatures with `ManifoldANN.jl` and
   write the matched-pair CSV (`benchmark_results/manl_validation_pairs.csv`):

   ```bash
   julia --project=. scripts/test_orcml_exact_match.jl
   ```

3. Render the three diagnostic plots (scatter, Bland-Altman,
   residual histogram) into `benchmark_results/`:

   ```bash
   cd benchmarking
   .venv/bin/python ../scripts/plot_orcml_validation.py
   ```

The scatter and Bland-Altman PDFs are the ones used in the thesis
figure; the residual histogram is included for completeness and not
shown in the thesis.

## Reproducing the bias diagnosis

The thesis attributes the residual `−0.037` bias to two specific
construction differences in the reference implementation rather than
to OT solver behaviour. The diagnostic that established this proceeds
as follows:

1. From `benchmark_results/manl_validation_pairs.csv` plus the two
   per-edge curvature CSVs (`curvatures_orcml_python.csv` and
   `curvatures_manl_exact_test.csv`), find the edge with the largest
   `|julia − python|`. In the seeded run above this is edge
   `(185, 247)` with `|Δ| ≈ 0.186`.

2. **Julia side.** Write a one-shot script that loads the dataset,
   builds the ORC-ManL graph the same way `test_orcml_exact_match.jl`
   does, and for the chosen edge dumps:

   - `mu_x.csv`  — neighbour ids and probabilities for the source
   - `mu_y.csv`  — neighbour ids and probabilities for the target
   - `C.csv`     — the cost matrix (rows = `mu_x` support,
                    cols = `mu_y` support)
   - `result.csv` — the `W₁` and `κ` reported by the Julia solver

   The relevant code is in `src/graphs/refinement/filtering.jl`,
   `solve_orc_for_edge!` (around lines 380–426).

3. **Python side.** Monkey-patch `GraphRicciCurvature` (in the venv
   under `benchmarking/.venv/lib/python3.13/site-packages/GraphRicciCurvature/`)
   so that the OT call captures the same three matrices for the same
   edge. The relevant function is
   `_compute_ricci_curvature_single_edge` and the densities come from
   `_get_single_node_neighbors_distributions`. Capture the same files
   under a separate output directory.

4. **Confirming it is not the solver.** Take each side's
   `(C, mu_x, mu_y)` and feed it to three independent OT solvers:

   - Julia: `OptimalTransport.jl` (network simplex via Tulip)
   - Python: `pot.emd2(mu_x, mu_y, C)`
   - Python: `scipy.optimize.linprog(...)` over the LP formulation

   All three solvers should agree to machine precision on each
   side's `W₁`. If they do, the OT solver is not the source of the
   bias.

5. **Confirming what *is* the source.** Diff `C^Julia` against
   `C^Python`, `mu_x^Julia` against `mu_x^Python`, and likewise for
   `mu_y`. The two differences observed in the published reference
   implementation are:

   - **`effective_eps` semantics differ.** Python's
     `compute_eff_eps_adj` does *not* exclude the edge endpoint from
     the neighbour list and discards the smallest distance via
     `argsort(dists)[1:k+1]`. Julia's `effective_epsilon` excludes
     the endpoint and keeps the smallest distance. This makes Python's
     ground cost matrix systematically larger on most edges, and the
     denominator in `κ = 1 − W₁/d` systematically larger as well.
   - **Source-vertex bug in target distribution.** The call
     `_get_single_node_neighbors_distributions(target, "successors")`
     binds the literal string `"successors"` to the `target` parameter
     of the function, so the source vertex is never removed from the
     target's neighbour list. The Julia implementation excludes both
     endpoints symmetrically.

6. **Sensitivity check.** For the worst-case edge, hold `W₁` fixed
   and swap only the denominator (Julia's vs. Python's). The induced
   change in `κ` is approximately `0.22`, which fully covers the
   observed worst-case `|Δ| = 0.186` and matches the value-dependent
   shape of the Bland-Altman residual.

The dump scripts used during the diagnostic were intentionally not
committed (they are one-shot diagnostic code rather than part of the
package surface); the steps above are sufficient to recreate them as
needed.

## Files involved

Committed to the repository:

- `scripts/generate_orcml_validation_data.py` — generates the swiss
  roll and reference curvatures via the orcml package.
- `scripts/test_orcml_exact_match.jl` — recomputes curvatures with
  `ManifoldANN.jl` and writes `manl_validation_pairs.csv`.
- `scripts/plot_orcml_validation.py` — produces the scatter,
  Bland-Altman, and residual-histogram plots.

Generated (under `benchmark_results/`, not committed):

- `test_data.csv`, `curvatures_orcml_python.csv`,
  `curvatures_manl_exact_test.csv`, `manl_validation_pairs.csv`
- `manl_validation_scatter.{png,pdf}`,
  `manl_validation_bland_altman.{png,pdf}`,
  `manl_validation_residual_hist.png`
