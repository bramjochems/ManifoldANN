# ManifoldANN.jl

Experimental Julia package for approximate nearest neighbours (ANN) on manifolds. The codebase favors modular indices, fast SIMD-friendly distance helpers, and graph materialization hooks for downstream manifold algorithms.

## Implemented indices

### Single-Level Indices
- `BruteForceIndex` – exact baseline with multithreaded scans
- `KDTreeIndex` – variance/cyclic split heuristics
- `LSHIndex` – multi-table random hyperplane + binned hashes
- `HNSWIndex` – configurable layer planner, neighbor policies, traversal policies
- `NNDescentIndex` – policy-driven NN-Descent graph builder with greedy graph search queries (see `docs/examples/indices/05-nndescent-index.jl`)

### Multi-Level Indices
- `MultiLevelIndex` – FAISS-like hierarchical indices with pluggable transforms, routing strategies, and merge policies
  - Supports IVF (Inverted File with Vector quantization) via `KMeansTransform`
  - Arbitrary nesting depth for multi-level hierarchies
  - Configurable routing strategies (`TopKRouting`, `ExhaustiveRouting`)
  - See `docs/examples/indices/06-ivf-index.jl` and `docs/transforms.md` for details

## Geometry and Geodesic Distances

- Local geometry is fitted with `PCAMethod` plus a neighborhood strategy via `LocalGeometryEstimator`. Strategies include `FixedNeighborhood`, shrinking `AdaptiveNeighborhood` (with `FitErrorCriterion` or `DistortionCriterion`), and growing `ExpandingNeighborhood` (with `DistortionCriterion` or `SubspaceAngleCriterion`), letting you tune the set of points used in each PCA tangent plane.
- Weighted graphs (`build_weighted_graph`) carry per-node tangent planes and edge weights computed with configurable modes: `SourceTangent` (fast, asymmetric), `SymmetricMean`, or `SymmetricMax`. Tangent planes can be unique per node (`NoSharing`) or reused when nearby planes are similar (`ShareSimilarTangents`).
- `build_geodesic_model` wraps an ANN index, weighted graph, and geometry method to answer geodesic distance queries. It supports node-to-node, point-to-node, and point-to-point queries by combining local tangent distances with Dijkstra shortest paths on the weighted graph.
- See runnable examples under `docs/examples/geodesic/` (e.g., `02-strategy-comparison.jl`) for end-to-end usage and strategy comparisons.

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
