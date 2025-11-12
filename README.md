# ManifoldANN.jl

Experimental Julia package for approximate nearest neighbours (ANN) on manifolds. The codebase favors modular indices, fast SIMD-friendly distance helpers, and graph materialization hooks for downstream manifold algorithms.

## Implemented indices

- `BruteForceIndex` – exact baseline with multithreaded scans
- `KDTreeIndex` – variance/cyclic split heuristics
- `LSHIndex` – multi-table random hyperplane + binned hashes
- `HNSWIndex` – configurable layer planner, neighbor policies, traversal policies
- `NNDescentIndex` – policy-driven NN-Descent graph builder with greedy graph search queries (see `docs/examples/indices/05-nndescent-index.jl`)

## Developing

There's make file to automate quick tasks

For running tests:
```bash
make test
```


For formatting code:

```bash
make format
```


## Directory Layout

- `src/` package source
- `test/unit/` unit tests, loaded via `test/runtests.jl`
- `docs/examples/` runnable documentation examples
- `docs/design/` architecture decision records (ADRs)
- `benchmarking/` Python<->Julia harness for comparing against ann-benchmarks baselines
