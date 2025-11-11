# ManifoldANN Benchmarking

Python wrapper for ManifoldANN with benchmarking against ann-benchmarks algorithms (hnswlib, Annoy).

## Quick Start

### Setup
```bash
cd benchmarking
./setup.sh  # Creates venv, installs dependencies
./fetch_ann_benchmarks.sh  # Clones pinned ann-benchmarks repo
source venv/bin/activate
```

### Run Benchmark
```bash
python benchmark.py
# or from repo root
make benchmark
```

Compares ManifoldANN (Julia) against hnswlib and Annoy on Fashion-MNIST.

## Files

- **`manifoldann_wrapper.py`** - Python↔Julia bridge using juliacall
  - Wraps: LSH, HNSW, KDTree, BruteForce
  - Supports batch queries for performance

- **`benchmark.py`** - Benchmark script
  - Tests ManifoldANN vs hnswlib vs Annoy
  - Measures QPS, recall@10, build time
  - Automatically uses batch queries for Julia algorithms
  - Compares both `neighbor_policy=heuristic` and `neighbor_policy=diversified` HNSW variants
- **`fetch_ann_benchmarks.sh`** - Clones the upstream ann-benchmarks repo (pinned commit)

- **`setup.sh`** - Automated setup (venv + dependencies)
- **`requirements.txt`** - Python dependencies
- **`data/`** - Cached datasets from ann-benchmarks

## Usage

### Different Dataset
```bash
python benchmark.py --dataset glove-25
python benchmark.py --dataset glove-50
```

### More Queries
```bash
python benchmark.py --n-queries 1000
```

### Full Dataset
```bash
python benchmark.py --max-train 0  # Uses full 60K training set
```

## Performance Features

1. **Batch Queries** - Process multiple queries in one Julia call (4x faster)
2. **Type-Stable Distance** - Distance functions stored in index
3. **Zero-Allocation Distance** - SIMD-optimized computations
4. **Multithreading Ready** - BruteForce uses `Threads.@threads`

## Output Example

```
============================================================
FINAL COMPARISON
============================================================
Algorithm                                                   Impl              Build(s)        QPS  Recall@10
-------------------------------------------------------------------------------------------------------------
⚪ HNSW(M=16, ef=50)                                        Other                0.45      7752.7     1.0000
🔵 HNSW(M=16, ef_construction=200, ef_search=50)           ManifoldANN           2.10       640.2     0.9640
🔵 LSH(n_tables=8, hash_length=16)                         ManifoldANN           0.38       190.1     0.4300
```

## Python↔Julia Overhead

The wrapper adds ~0.02ms overhead **per query** due to language boundary crossing:
- Array conversion (numpy → Julia, Fortran layout)
- Function call across language boundary
- Result conversion and index translation (0-based ↔ 1-based)

**Impact by query time:**
- Query takes 0.01ms → 3x slowdown (overhead dominates)
- Query takes 1ms → 1.02x slowdown (overhead negligible)
- Query takes 100ms → 1.0002x slowdown (overhead irrelevant)

**Mitigations:**
- ✅ Batch queries reduce overhead by ~4x (implemented)
- ✅ Type converters pre-created (not eval'd per call)
- ✅ SIMD-optimized distance functions reduce query time

For most benchmarking use cases (query time >1ms), overhead is acceptable (1-2%).

## Notes

- **hnswlib is multithreaded** (uses all CPU cores)
- **ManifoldANN is currently single-threaded** (multithreading planned)
- Datasets cached in `data/` after first download

## Available Datasets

| Dataset | Dimensions | Train | Test | Size | Metric |
|---------|-----------|-------|------|------|--------|
| fashion-mnist | 784 | 60K | 10K | ~30MB | Euclidean |
| glove-25 | 25 | 1.18M | 10K | ~120MB | Angular |
| glove-50 | 50 | 1.18M | 10K | ~240MB | Angular |

## Troubleshooting

### "Package ManifoldANN not found"
```bash
cd /path/to/ManifoldANN
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### First query very slow
Julia uses JIT compilation. First query triggers compilation (~seconds), subsequent queries are fast.

### Memory errors
Large datasets require more RAM. Monitor with:
```bash
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().available / 1e9:.1f} GB')"
```

## Requirements

- Python 3.8+
- Julia 1.6+ with ManifoldANN installed
- ~500MB disk space for datasets

## References

- [ann-benchmarks](https://github.com/erikbern/ann-benchmarks) (commit: `f402b2cc17b980d7cd45241ab5a7a4cc0f965e55`)
- [juliacall documentation](https://juliapy.github.io/PythonCall.jl/stable/juliacall/)
