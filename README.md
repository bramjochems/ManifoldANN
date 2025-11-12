# ManifoldANN.jl

Experimental Julia package for approximate nearest neighbours (ANN) on manifolds. The codebase favors modular indices, fast SIMD-friendly distance helpers, and graph materialization hooks for downstream manifold algorithms.

## Implemented indices

- `BruteForceIndex` – exact baseline with multithreaded scans
- `KDTreeIndex` – variance/cyclic split heuristics
- `LSHIndex` – multi-table random hyperplane + binned hashes
- `HNSWIndex` – configurable layer planner, neighbor policies, traversal policies
- `NNDescentIndex` – policy-driven NN-Descent graph builder with greedy graph search queries (see `docs/examples/indices/05-nndescent-index.jl`)

## Developing

There's a Makefile to automate common tasks:

### Running Tests
```bash
make test
```

### Formatting Code
```bash
make format
```

### Benchmarking

First-time setup (creates venv, installs dependencies):
```bash
make benchmark-setup
```

Run benchmarks:
```bash
make benchmark ARGS="fashion-mnist"
make benchmark ARGS="nytimes -k 20"
make benchmark ARGS="--list-configs"
```

The benchmark target automatically uses the virtual environment if available.

See [`benchmarking/README.md`](benchmarking/README.md) and [`benchmarking/REFACTOR_SUMMARY.md`](benchmarking/REFACTOR_SUMMARY.md) for detailed benchmarking documentation.


## Directory Layout

- `src/` - Julia package source code
- `test/` - Unit tests (`test/runtests.jl` entry point)
- `docs/examples/` - Runnable documentation examples
- `docs/design/` - Architecture decision records (ADRs)
- `benchmarking/` - Benchmarking suite for ANN library comparison
  - `configs/` - Per-dataset YAML configurations
  - `benchmarking/` - Python package (utils, wrappers, registry)
  - `julia/` - Separate Julia environment for Julia library benchmarks
- `scripts/` - Development and testing scripts
