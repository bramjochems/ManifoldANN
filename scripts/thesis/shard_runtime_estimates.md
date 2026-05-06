# Shard runtime estimates (revised)

Corrections from the first draft:
- MANN-HNSW is now **~2× slower than HNSWlib**, not faster (we never said it was; was a mis-estimate).
- Added Tier 1 mann-nnd-sweep on mid-large datasets.
- Added Tier 1 mann-ivf-flat / mann-ivf-hnsw on large datasets.
- Tier 2 NND.jl and HNSW.jl shards: SLOW (single-threaded for HNSW.jl, slow for NND.jl per shard).

Reference per-build times from existing thesis CSVs (laptop, 16 threads):

| Dataset (n × d) | HNSWlib | MANN-HNSW (=2× HNSWlib) | LSH | FAISS-IVF | Annoy | NND-deferred | RPForest |
|---|---|---|---|---|---|---|---|
| fashion-mnist 60k×784 | 8s | 16s | 1s | 1s | 2s | 8s | 1s |
| mnist 60k×784 | 8s | 16s | 1s | 1s | 2s | 8s | 1s |
| lastfm 290k×65 | 6s | 12s | 2s | <1s | 2s | 60s | 2s |
| nytimes 290k×256 | 43s | 86s | 4s | <1s | 3s | 60s | 3s |
| glove-25 1.2M×25 | 47s | 94s | 5s | <1s | 3s | 250s | 3s |
| glove-50 1.2M×50 | 69s | 138s | 5s | <1s | 4s | 270s | 4s |
| glove-100 1.2M×100 | 143s | 286s | 6s | 1s | 9s | 320s | 5s |
| sift 1M×128 | 84s | 168s | 4s | <1s | 6s | 96s | 4s |
| gist 1M×960 | est 500s | est 1000s | 10s | est 5s | 30s | est 800s | 6s |

Build counts per shard:
- mann-hnsw / hnswlib sweep: **6 builds**
- mann-lsh sweep: **12 builds**
- faiss-ivf sweep: **3 builds**
- annoy sweep: **4 builds**
- mann-rpforest sweep: **9 builds**
- mann-nnd sweep: **9 builds** (k × max_iterations × max_candidates → 3×3 builds × 3 max_candidates) — but max_candidates is query-time so 9 actual builds
  Wait correction: build sweep is k × max_iterations = 3 × 1 = 3 builds (using max_iter=15 only) or 3 × 2 = 6 if we sweep max_iter. Let's say **6 builds** for mann-nnd-sweep.
- mann-ivf-flat sweep: **4 builds** (nlist), **7 query points** (nprobe)
- mann-ivf-hnsw sweep: **9 builds** (nlist × ef_search baked in)
- single-point: **6 single builds** (BruteForce, KDTree, SciPy-KDTree, NND, PyNNDescent, [+IVF-HNSW/IVF-Flat for small datasets])
- Tier 2 hnsw-headtohead: **12 builds total** (6 MANN + 6 HNSW.jl single-threaded)
- Tier 2 nnd-headtohead: **12 builds total** (6 MANN + 6 NND.jl)
- Tier 2 nnd-variants: **24 builds** (6 builds × 4 variants)

## Per-shard runtime estimates

Build dominates. Multiply per-build × n_builds. Add ~1-3 min for warmup, JIT, single-point-method overhead. Plus ~5 min per VM startup (paid once per VM, not per shard).

### fashion-mnist (60k, 784d, euclidean) — small, all fast

| Shard | Runtime |
|---|---|
| fashion-mnist-mann-hnsw-sweep | 3 min |
| fashion-mnist-hnswlib-sweep | 2 min |
| fashion-mnist-mann-lsh-sweep | 1 min |
| fashion-mnist-faiss-ivf-sweep | 1 min |
| fashion-mnist-annoy-sweep | 1 min |
| fashion-mnist-mann-rpforest-sweep | 1 min |
| fashion-mnist-single-point-methods | 3 min |
| **Total** | **12 min** |

### mnist (60k, 784d, euclidean)
Same as fashion-mnist: **~12 min**

### lastfm (290k, 65d, dot)

| Shard | Runtime |
|---|---|
| lastfm-mann-hnsw-sweep | 3 min |
| lastfm-hnswlib-sweep | 2 min |
| lastfm-mann-lsh-sweep | 1 min |
| lastfm-faiss-ivf-sweep | 1 min |
| lastfm-annoy-sweep | 1 min |
| lastfm-mann-rpforest-sweep | 1 min |
| lastfm-mann-nnd-sweep | 7 min (6 × 60s) |
| lastfm-single-point-methods | 5 min |
| **Total** | **21 min** |

### nytimes (290k, 256d, angular)

| Shard | Runtime |
|---|---|
| nytimes-mann-hnsw-sweep | 10 min (6 × 86s) |
| nytimes-hnswlib-sweep | 5 min (6 × 43s) |
| nytimes-mann-lsh-sweep | 1 min |
| nytimes-faiss-ivf-sweep | 1 min |
| nytimes-annoy-sweep | 1 min |
| nytimes-mann-rpforest-sweep | 1 min |
| nytimes-mann-nnd-sweep | 7 min (6 × 60s) |
| nytimes-single-point-methods | 4 min |
| nytimes-tier2-hnsw-headtohead | **65 min** (HNSW.jl 6 × 540s = 54 min + MANN 6 × 86s = 9 min) ★ slow |
| **Total** | **95 min** |

### glove-25 (1.2M, 25d, angular)

| Shard | Runtime |
|---|---|
| glove-25-mann-hnsw-sweep | 10 min (6 × 94s) |
| glove-25-hnswlib-sweep | 5 min |
| glove-25-mann-lsh-sweep | 1 min |
| glove-25-faiss-ivf-sweep | 1 min |
| glove-25-annoy-sweep | 1 min |
| glove-25-mann-rpforest-sweep | 1 min |
| glove-25-mann-nnd-sweep | 25 min (6 × 250s) |
| glove-25-single-point-methods | 6 min |
| glove-25-tier2-nnd-headtohead | **70 min** (NND.jl 6 × 400s = 40 min + MANN 6 × 250s = 25 min + setup) ★ slow |
| glove-25-tier2-nnd-variants | **100 min** (24 × 250s = 100 min) ★★ slow |
| **Total** | **220 min** |

### glove-50 (1.2M, 50d, angular)

| Shard | Runtime |
|---|---|
| glove-50-mann-hnsw-sweep | 14 min (6 × 138s) |
| glove-50-hnswlib-sweep | 7 min |
| glove-50-mann-lsh-sweep | 1 min |
| glove-50-faiss-ivf-sweep | 1 min |
| glove-50-annoy-sweep | 1 min |
| glove-50-mann-rpforest-sweep | 1 min |
| glove-50-mann-nnd-sweep | 27 min (6 × 270s) |
| glove-50-single-point-methods | 7 min |
| **Total** | **59 min** |

### glove-100 (1.2M, 100d, angular)

| Shard | Runtime |
|---|---|
| glove-100-mann-hnsw-sweep | 29 min (6 × 286s) |
| glove-100-hnswlib-sweep | 15 min |
| glove-100-mann-lsh-sweep | 1 min |
| glove-100-faiss-ivf-sweep | 1 min |
| glove-100-annoy-sweep | 2 min |
| glove-100-mann-rpforest-sweep | 1 min |
| glove-100-mann-nnd-sweep | 32 min (6 × 320s) |
| glove-100-mann-ivf-flat-sweep | 5 min |
| glove-100-mann-ivf-hnsw-sweep | 30 min (9 builds) |
| glove-100-single-point-methods | 10 min (now without IVF-Flat / IVF-HNSW) |
| **Total** | **126 min** |

### sift (1M, 128d, euclidean)

| Shard | Runtime |
|---|---|
| sift-mann-hnsw-sweep | 17 min (6 × 168s) |
| sift-hnswlib-sweep | 9 min |
| sift-mann-lsh-sweep | 1 min |
| sift-faiss-ivf-sweep | 1 min |
| sift-annoy-sweep | 1 min |
| sift-mann-rpforest-sweep | 1 min |
| sift-mann-nnd-sweep | 10 min (6 × 96s) |
| sift-mann-ivf-flat-sweep | 5 min |
| sift-mann-ivf-hnsw-sweep | 25 min (9 builds) |
| sift-single-point-methods | 8 min |
| **Total** | **78 min** |

### gist (1M, 960d, euclidean) — slowest dataset

| Shard | Runtime |
|---|---|
| gist-mann-hnsw-sweep | **100 min** (6 × 1000s) ★★ |
| gist-hnswlib-sweep | **50 min** (6 × 500s) ★ |
| gist-mann-lsh-sweep | 2 min |
| gist-faiss-ivf-sweep | 1 min |
| gist-annoy-sweep | 3 min |
| gist-mann-rpforest-sweep | 1 min |
| gist-mann-ivf-flat-sweep | 8 min |
| gist-mann-ivf-hnsw-sweep | 30 min |
| gist-single-point-methods | 30 min (KDTree slow on 1M × 960d) |
| **Total** | **225 min** |

## Long poles (>30 min, fail-fast → run last on their VM)

| # | Shard | Min |
|---|---|---|
| 1 | glove-25-tier2-nnd-variants | 100 |
| 2 | gist-mann-hnsw-sweep | 100 |
| 3 | glove-25-tier2-nnd-headtohead | 70 |
| 4 | nytimes-tier2-hnsw-headtohead | 65 |
| 5 | gist-hnswlib-sweep | 50 |
| 6 | glove-100-mann-nnd-sweep | 32 |
| 7 | gist-single-point-methods | 30 |
| 8 | glove-100-mann-ivf-hnsw-sweep | 30 |
| 9 | glove-100-mann-hnsw-sweep | 29 |
| 10 | glove-50-mann-nnd-sweep | 27 |
| 11 | sift-mann-ivf-hnsw-sweep | 25 |
| 12 | glove-25-mann-nnd-sweep | 25 |

## Total work

| Dataset | Min |
|---|---|
| fashion-mnist | 12 |
| mnist | 12 |
| lastfm | 21 |
| nytimes | 95 |
| glove-25 | 220 |
| glove-50 | 59 |
| glove-100 | 126 |
| sift | 78 |
| gist | 225 |
| **Total** | **848 min ≈ 14 h** |

With 10 VMs, target = **~85 min/VM**.

But the floor is **`gist-mann-hnsw-sweep` (100 min) on its own VM**, plus other slow shards on other VMs. Realistic floor: **~100 min** (~1h40 wall-clock).
