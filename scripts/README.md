# Scripts

Three buckets:

- `thesis/` — produces results and figures for the thesis. Reads from
  `<package>/docs/examples/...` and writes to
  `<parent>/mai/thesis/docs/thesis/{results,figures}` (the thesis docs tree
  outside this repo). Long-term home is the sibling thesis-code repo per
  `CLAUDE.md`'s separation principle.
- `perf/` — focused fair-compare scripts that produce thesis-grade head-to-head
  numbers (recall-vs-qps Pareto, single-config head-to-head). Holding pen until
  `benchmarking/` migrates out; then these go with it.
- `archive/` — superseded one-off benchmarks, profile drivers, and diagnostic
  scripts. Kept in case the underlying question comes back.

Correctness assertions (regression / unit) live under `test/`, not here.

## thesis/

### orc/ — Ollivier-Ricci curvature on synthetic manifolds

```bash
# Full pipeline (swiss roll + torus + analysis)
./scripts/thesis/orc/run_all.sh

# Smoke test
SMOKE=1 ./scripts/thesis/orc/run_all.sh

# Phase-specific
./scripts/thesis/orc/run_detection.sh        # SKIP_PRUNING=1
./scripts/thesis/orc/run_pruning.sh          # SKIP_DETECTION=1

# Sampling diagnostic
./scripts/thesis/orc/run_orc_sampling_diagnostic.sh
```

Resume an interrupted run by setting `RESUME_DIR=<path-to-prior-output>`.

| File | Description |
|---|---|
| `experiment_orc.jl` | Unified swiss-roll + torus experiment (driven by `MANIFOLD={swiss,torus}`) |
| `experiment_orc_sampling_diagnostic.jl` | Non-uniform sampling diagnostic |
| `orc_helpers.jl` | Shared evaluation logic + CSV I/O |
| `analyze_orc.py` | Aggregate detection/pruning CSVs + emit thesis figures |
| `analyze_orc_benchmarks.jl` | Solver-comparison analysis |
| `plot_sampling_diagnostic.py` | Visualisation for the sampling diagnostic |
| `run_*.sh` | Shell wrappers |

Environment overrides: `SMOKE`, `RESUME_DIR`, `N_OVERRIDE`, `K_OVERRIDE`,
`VARIANT_OVERRIDE`, `SKIP_EDGES`, `SKIP_GEODERROR`, `SKIP_PRUNING`,
`SKIP_DETECTION`, `JULIA_NUM_THREADS`.

### composite/ — Composite shortcut labelling study

Validates the composite (chord-ratio AND graph-effect) shortcut label against
chord-only on three pre-existing manifold runs.

| File | Description |
|---|---|
| `composite_shortcut_validation.py` | Validation against three manifold cells |
| `composite_shortcut_full_eval.py` | Full eval (~8 min) |
| `composite_shortcut_extended_eval.py` | Extended pruning sweep |
| `composite_shortcut_dump_points.jl` | Reproduce point clouds (RNG-pinned) |
| `_composite_full_eval_gen_points.jl` | Helper invoked by full_eval |
| `all_pairs_mre_swiss.py` | All-pairs MRE on swiss roll |
| `plot_composite_auroc.py` / `plot_composite_coverage.py` | Figures |

### geodesic/ — Geodesic estimation end-to-end

| File | Description |
|---|---|
| `experiment_geodesic_e2e.jl` | E2E geodesic estimation experiment |
| `plot_geodesic_e2e.py` | MRE-vs-n and decomposition figures |

### orcml/ — Validation against the reference Python `orcml`

| File | Description |
|---|---|
| `orcml_validation.md` | Walk-through |
| `generate_orcml_validation_data.py` | Generate swiss-roll inputs |
| `plot_orcml_validation.py` | Scatter / Bland-Altman figures |
| `extract_edge_curvatures.py` | Pull per-edge curvatures from results |
| `test_orcml_exact_match.jl` | Validates Julia ORC matches Python orcml on a fixed input |
| `test_orcml_exact_match_policy.jl` | Same, with the `OrcmlExact` compatibility profile |
| `orcml_torus_flag.py` | Torus-flag exploratory script |

## perf/

Focused fair-compare scripts. Apples-to-apples library comparisons happen on
the recall-vs-qps Pareto curve, not at fixed parameter values; see
`nndescent_jl_pareto.jl` for the canonical pattern.

| File | Description |
|---|---|
| `hnsw_fair_compare.py` | HNSW: ManifoldANN vs hnswlib vs HNSW.jl |
| `kdtree_fair_compare.jl` | KDTree: ManifoldANN vs NearestNeighbors.jl |
| `nndescent_jl_pareto.jl` | NN-Descent: recall-vs-qps Pareto sweep |
| `nndescent_jl_compare.jl` | Single-point head-to-head |

## archive/

Superseded one-shots: per-edge microbenches, build/query profile drivers,
diagnostics from past refactors, the early ORC swiss/torus scripts that the
unified `experiment_orc.jl` replaced.
