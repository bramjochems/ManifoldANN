# Julia Threading Configuration for Benchmarks

## Overview

ManifoldANN uses Julia's multi-threading capabilities to parallelize index construction, particularly for multi-level indices like IVF-HNSW. The benchmark suite now automatically configures Julia threading for optimal performance.

## Automatic Configuration

The benchmark script (`benchmark.py`) automatically detects your CPU core count and configures Julia threading accordingly:

```bash
python benchmark.py fashion-mnist
```

You'll see:
```
⚙️  Auto-configured Julia threading: JULIA_NUM_THREADS=16
...
✓ Julia threading enabled: 16 threads
```

## Manual Configuration

To manually control the number of threads:

### Linux/WSL
```bash
export JULIA_NUM_THREADS=$(nproc)
python benchmark.py fashion-mnist
```

### macOS
```bash
export JULIA_NUM_THREADS=$(sysctl -n hw.ncpu)
python benchmark.py fashion-mnist
```

### Specific thread count
```bash
export JULIA_NUM_THREADS=8
python benchmark.py fashion-mnist
```

## Performance Impact

Threading significantly improves build times for multi-level indices:

| Algorithm | Build Strategy | Without Threading | With Threading (16 cores) | Speedup |
|-----------|----------------|-------------------|---------------------------|---------|
| HNSW | Single index | 8.5s | 8.5s | 1.0x |
| IVF-HNSW | 100 parallel indices | ~14s | ~9-10s | 1.4-1.5x |

### Where Threading Helps

**IVF-HNSW** builds 100 separate HNSW indices in parallel (one per cluster):
- **Without threading**: Sequential, ~5-6s overhead
- **With threading**: Parallel across CPU cores, minimal overhead

### Where Threading Doesn't Help (Yet)

- **KMeans clustering** (~7.4s): Not yet parallelized
- **Single index builds** (HNSW, KD-Tree): Already optimized

## Technical Details

### Why Thread Configuration Matters

Julia's threading must be set **before** the Julia runtime initializes. The benchmark script handles this automatically by:

1. Setting `JULIA_NUM_THREADS` at the very top of `benchmark.py`
2. Before any imports that could trigger Julia initialization
3. Verifying thread count after Julia loads

### Verification

The benchmark script automatically verifies threading is working:

```python
✓ Julia threading enabled: 16 threads
```

If you see a warning:
```
⚠️  WARNING: Julia is using only 1 thread!
```

This means threading couldn't be configured (e.g., Julia was already initialized by another script). Restart your Python session.

## Troubleshooting

### "Julia is using only 1 thread"

**Cause**: Julia runtime was already initialized before `JULIA_NUM_THREADS` was set.

**Solution**: Start a fresh Python session:
```bash
# Exit Python/Jupyter
exit()  # or Ctrl+D

# Set env variable, then run benchmark
export JULIA_NUM_THREADS=16
python benchmark.py fashion-mnist
```

### "Slower than expected IVF build times"

**Check actual thread usage**:
```bash
python -c "from juliacall import Main; print('Threads:', Main.seval('Threads.nthreads()'))"
```

Expected: Should match CPU count
If you see `1`: Threading not configured

### Performance Still Slow?

**Test your configuration**:
```bash
cd benchmarking
source venv/bin/activate
python test_threading_impact.py
```

Expected output:
```
HNSW (single):     3.28s
IVF-HNSW (100):    3.56s
Ratio:             1.09x
✓ GOOD: IVF-HNSW build time is reasonable
```

If ratio > 2.0x, threading is not working properly.

## Implementation Notes

The threading configuration is implemented in three places:

1. **`benchmark.py`** (lines 9-25): Sets environment variable before imports
2. **`_verify_julia_threading()`** (lines 72-104): Verifies actual thread count
3. **`builder.jl`** (lines 122-127): Uses `Threads.@threads` for parallel builds

This ensures:
- Threading is configured as early as possible
- Users are warned if threading fails
- Parallel algorithms automatically use all available threads
