# Julia Benchmarking Environment

Separate Julia environment for benchmarking ManifoldANN against other Julia ANN libraries.

## Purpose

This environment provides a clean separation between:
- **ManifoldANN** - The main library (clean, no benchmark dependencies)
- **Benchmark dependencies** - Comparison libraries like NearestNeighbors.jl, HNSW.jl

This follows the separation principle from CLAUDE.md and mirrors the Python benchmarking package structure.

## Included Libraries

- **ManifoldANN** - Local dev dependency (from `../..`)
- **NearestNeighbors.jl** - KD-tree, Ball tree, brute force implementations
- **HNSW.jl** - Pure Julia HNSW implementation

## Setup

The Julia environment is automatically set up when running:

```bash
make benchmark-setup
```

Or manually:

```bash
cd benchmarking/julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Adding New Julia Libraries

1. Add to `Project.toml`:
   ```toml
   [deps]
   NewLibrary = "uuid-here"
   ```

2. Create wrapper in `wrappers/newlibrary.jl`

3. Register in Python registry (when implementing Python<->Julia bridge)

## Wrappers

The `wrappers/` directory will contain Julia code that provides a consistent interface for different Julia ANN libraries, making them callable from the Python benchmark harness via `juliacall`.

## Manifest.toml

The `Manifest.toml` should be **committed** to ensure reproducible benchmarks with exact dependency versions.
