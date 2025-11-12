# Development Scripts

This directory contains one-off development and testing scripts.

## Files

- `benchmark_nndescent.jl` - Performance testing for NN-Descent implementation
- `benchmark_reference.jl` - Reference comparison with NearestNeighborDescent.jl

## Usage

These are standalone Julia scripts for development purposes:

```bash
julia --project=. scripts/benchmark_nndescent.jl
julia --project=. scripts/benchmark_reference.jl
```

## Note

For comprehensive benchmarking against multiple ANN libraries, use the Python benchmarking suite in `benchmarking/`:

```bash
make benchmark-setup
make benchmark ARGS="fashion-mnist"
```
