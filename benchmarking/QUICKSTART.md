# Quick Start Guide

## Setup (One-time)

```bash
cd benchmarking
./setup.sh  # Automated setup
./fetch_ann_benchmarks.sh
source venv/bin/activate
```

Or manually:
```bash
cd benchmarking
python3 -m venv venv
source venv/bin/activate
pip install juliacall numpy h5py
./fetch_ann_benchmarks.sh

cd ..
julia --project=. -e 'using Pkg; Pkg.instantiate()'
cd benchmarking
```

## Run Benchmark

```bash
source venv/bin/activate
python benchmark.py
```

**Expected output:**
- Algorithm comparison table (QPS, recall@10)
- Runtime: ~1-2 minutes for 100 queries on Fashion-MNIST
- First run is slower due to Julia JIT compilation

## Try Different Options

```bash
# Different dataset
python benchmark.py --dataset glove-25
python benchmark.py --dataset glove-50

# More queries for better statistics
python benchmark.py --n-queries 1000

# Full training set (slower)
python benchmark.py --max-train 0
```

## Common Issues

**"Package ManifoldANN not found"**
```bash
cd /path/to/ManifoldANN
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**"Module juliacall not found"**
```bash
source venv/bin/activate
pip install juliacall
```

**First query very slow (>1 second)**
- Normal! Julia JIT compilation on first run
- Subsequent queries are fast
- Test script includes warmup to avoid timing this

**Memory errors**
```bash
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / 1e9:.1f} GB')"
```
Large datasets (GloVe-50+) may need 8GB+ RAM.

## Understanding Results

**Recall@10**: Fraction of true nearest neighbors found (higher is better, 1.0 = perfect)

**QPS**: Queries per second (higher is better)

**Notes:**
- hnswlib uses multithreading (all cores), ManifoldANN is single-threaded
- Python↔Julia wrapper adds ~0.02ms overhead per query (negligible for most cases)
- Batch queries reduce overhead by ~4x (automatically used)

See [README.md](README.md) for full documentation.
