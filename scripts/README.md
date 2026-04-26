# Scripts

## ORC Experiments

Evaluate Ollivier-Ricci curvature as a shortcut detector on synthetic manifolds.

### Running

```bash
# Full runs (each creates a timestamped output directory)
./scripts/run_orc_swiss_roll.sh
./scripts/run_orc_torus.sh
./scripts/run_orc_sampling_diagnostic.sh

# Smoke tests (single small config, fast)
SMOKE=1 ./scripts/run_orc_swiss_roll.sh

# Run all experiments sequentially
./scripts/run_orc_all.sh
```

### Resuming interrupted runs

Results are written incrementally after each ORC computation. If a run is
interrupted, set `RESUME_DIR` to the output directory to skip already-computed
configs:

```bash
RESUME_DIR=.../orc_results/swiss_roll_20260221_143012 ./scripts/run_orc_swiss_roll.sh
```

### Environment variables

| Variable | Scripts | Description |
|---|---|---|
| `SMOKE=1` | all | Minimal single-config run |
| `RESUME_DIR=path` | all | Resume into a previous run's directory |
| `N_OVERRIDE=500` | swiss roll, torus | Single n value |
| `K_OVERRIDE=10` | swiss roll, torus | Single k value |
| `VARIANT_OVERRIDE=R2r1` | torus | Single torus geometry variant |
| `SKIP_EDGES=1` | swiss roll, torus | Skip per-edge CSV (saves time) |
| `SKIP_GEODERROR=1` | swiss roll, torus | Skip geodesic error analysis |

### Output structure

Each run creates a timestamped directory under `docs/thesis/results/orc_results/`:

```
swiss_roll_20260221_143012/
  raw.csv          # one row per (n, k, noise, variant, tau)
  pivot_f1.csv     # F1 at kappa=0 pivoted (rows=n, cols=k)
  pivot_best.csv   # best-threshold F1 pivoted
  edges.csv        # per-edge data for AUROC (optional)
  geoderror.csv    # geodesic error at pruning thresholds (optional)
  config.toml      # git hash + parameters
```

### Script files

| File | Description |
|---|---|
| `experiment_orc_swiss_roll.jl` | Swiss roll experiment |
| `experiment_orc_torus.jl` | Torus experiment (3 geometry variants) |
| `experiment_orc_sampling_diagnostic.jl` | Non-uniform sampling diagnostic |
| `orc_helpers.jl` | Shared evaluation logic, CSV I/O helpers |
| `plot_sampling_diagnostic.py` | Visualisation for sampling diagnostic |
| `run_orc_*.sh` | Shell wrappers |

## Benchmarks

- `benchmark_nndescent.jl` - Performance testing for NN-Descent
- `benchmark_reference.jl` - Reference comparison with NearestNeighborDescent.jl

```bash
julia --project=. scripts/benchmark_nndescent.jl
```

For the full benchmarking suite, see `benchmarking/`.
