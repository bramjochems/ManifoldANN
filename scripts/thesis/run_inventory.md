# Thesis benchmark run inventory

## Target VM
- **VM size**: `Standard_D16s_v6` (Intel Xeon Platinum 8573C / Emerald Rapids, 16 vCPU, 64 GB RAM, AVX-512 + AMX)
- **Region**: West Europe
- **Threads**: 16 (matches vCPU count)
- **Cost**: ~$0.65/hour billed per second

## Reps policy
- **Tier 1 (baseline sweep)**: 5 reps per (build, query_combo) — recall stable but query timing has noise we want to average
- **Tier 2 (Julia head-to-heads)**: 3 reps — build dominates wall-clock and recall is what matters

## Cost model
- **Build time dominates.** A single build can take 1-30 minutes depending on (dataset, method, params). Once an index is built, query sweeps are extremely cheap (seconds per combo).
- **Over-collect query points.** Query points are dropped in post-processing if not useful. Re-running because we missed a Pareto point is expensive.
- **5 reps × N query points adds negligible time** — query loop runs in seconds even with N=10 sweep points.

## Runtime baseline (from existing thesis results)
All timings on user's laptop (16 threads). The D16s_v6 (Emerald Rapids) should be modestly faster (maybe 1.2-1.5×). Conservative estimates below use the laptop figures.

---

## Tier 1: baseline sweep across all 8 datasets

### Dataset list
| Dataset            | n_train   | dim | metric    |
|--------------------|-----------|-----|-----------|
| fashion-mnist      | 60,000    | 784 | euclidean |
| mnist              | 60,000    | 784 | euclidean |
| lastfm             | 292,385   | 65  | dot       |
| glove-25           | 1,183,514 | 25  | angular   |
| glove-50           | 1,183,514 | 50  | angular   |
| glove-100          | 1,183,514 | 100 | angular   |
| nytimes            | 290,000   | 256 | angular   |
| sift               | 1,000,000 | 128 | euclidean |
| gist               | 1,000,000 | 960 | euclidean |

### Methods (Tier 1)
- **MANN-BruteForce** (small datasets only — sanity check)
- **MANN-KDTree**, **SciPy-KDTree** (Euclidean datasets only)
- **MANN-LSH** — sweep n_tables × hash_length
- **MANN-HNSW-diversified** — build sweep (M, ef_c) × query sweep (ef_search)
- **MANN-IVF-HNSW** — single point per dataset (ef_search baked in)
- **MANN-NNDescent-full-deferred** — single sweep candidate (best of the NND variants based on existing results)
- **MANN-RPForest** — NEW. Build sweep (n_trees, leaf_cap) × query sweep (search_k or similar). Wrapper to be written.
- **HNSWlib** — build sweep (M, ef_c) × query sweep (ef_search)
- **FAISS-IVF** — build sweep (nlist) × query sweep (nprobe)
- **Annoy** — build sweep (n_trees) × query sweep (search_k)
- **PyNNDescent** — single point per dataset (or small sweep on n_neighbors)

### Methods explicitly DROPPED from Tier 1
- **HNSW-jl** — too slow (1200-3300s per build), no query-time params worth sweeping. Goes to Tier 2.
- **NearestNeighborDescent-jl** — slow + already a single-point method. Goes to Tier 2.
- **MANN-NNDescent-pruned-* / MANN-NNDescent-full-continuous** — keep one NND variant only (full-deferred is the cleanest based on existing results). Reduces clutter.

### Build/query sweep grids (Tier 1)

#### MANN-HNSW-diversified
- Build configs: M ∈ {8, 16, 32}, ef_c ∈ {100, 200} → 6 builds per dataset
- Query sweep: ef_search ∈ {16, 32, 64, 100, 150, 200, 300, 400} → 8 query points per build
- Total per dataset: 6 builds × 8 query points = **48 measurements** (post-processing drops what's not useful)

#### HNSWlib
- Build configs: M ∈ {8, 16, 32}, ef_c ∈ {100, 200} → 6 builds
- Query sweep: ef_search ∈ {16, 32, 64, 100, 150, 200, 300, 400} → 8 query points
- Total per dataset: **48 measurements**

#### MANN-LSH
- Build sweep: n_tables ∈ {5, 10, 20, 40}, hash_length ∈ {8, 12, 16} → 12 builds
- No query sweep
- Total per dataset: **12 measurements**

#### FAISS-IVF
- Build sweep: nlist ∈ {64, 256, 1024} → 3 builds
- Query sweep: nprobe ∈ {1, 2, 4, 8, 16, 32, 64} → 7 query points
- Total per dataset: **21 measurements**

#### Annoy
- Build sweep: n_trees ∈ {10, 50, 100, 200} → 4 builds
- Query sweep: search_k ∈ {-1 (default), 100, 500, 1000, 5000, 10000} → 6 query points
- Total per dataset: **24 measurements**

#### MANN-RPForest (NEW — wrapper TBD)
- Build sweep: n_trees ∈ {5, 10, 20}, leaf_cap ∈ {32, 64, 128} → 9 builds
- Query sweep: TBD based on the actual implementation knobs (likely 4-6 points)
- Total per dataset: ~**36-54 measurements**

#### Single-point methods (no sweep)
- MANN-BruteForce, MANN-KDTree, SciPy-KDTree, MANN-IVF-HNSW, MANN-NNDescent-full-deferred, PyNNDescent

### Estimated runtime per dataset (Tier 1)

Build time dominates. Sum of (n_builds × est_build_time) across all sweeping methods on each dataset, plus a small allowance for single-point methods. Existing thesis CSVs were the input.

Per-dataset estimates (roughly, on D16s_v6 / 16 threads):

| Dataset       | Est. wall-clock | Dominant cost |
|---------------|-----------------|---------------|
| fashion-mnist | 10-15 min       | All builds <1 min. HNSW × 6 builds × 1-2 min each. |
| mnist         | 10-15 min       | Same. |
| lastfm        | 30-45 min       | HNSW × 6 builds × 2-3 min. |
| nytimes       | 1-1.5 h         | HNSW × 6 builds × ~5-10 min each on heavy configs. |
| glove-25      | 1-1.5 h         | MANN-HNSW build was 1275s on heaviest config. |
| glove-50      | 1.5-2 h         | Similar but heavier per build. |
| glove-100     | 2-3 h           | MANN-HNSW build was 2178s on heaviest. 6 builds × ~10-20 min. |
| sift          | 1-1.5 h         | MANN-HNSW build was 80s. KDTree slow but only one config. |
| gist          | 4-6 h           | 960d slow distances. HNSW × 6 builds × 30-50 min each likely. |

**Tier 1 total sequential: ~12-18 hours.** Trivially parallelisable across datasets.

---

## Tier 2: Julia head-to-heads

### Goal
Show MANN's Julia implementations are competitive with native Julia ANN packages. NOT to compete with C++ libraries (that's Tier 1).

### Comparisons

#### MANN-HNSW vs HNSW.jl on **nytimes** (single dataset)
- Configs: M ∈ {8, 12, 16}, ef_c ∈ {100, 200} → 6 builds per library
- Query sweep: ef_search ∈ {25, 50, 100, 200} → 4 query points per build
- Reps: 3
- Two libraries × 6 builds × 4 query points = 48 measurements
- **Why nytimes**: HNSW.jl took 539s on nytimes vs 1200-3300s on glove and SIFT. Tractable.
- **Estimated runtime**: HNSW.jl is single-threaded — 6 × 540s = ~55 min for HNSW.jl alone, plus ~10 min for MANN-HNSW. **~1.5 h total.**

#### MANN-NND vs NND.jl on **glove-25** (single dataset)
- Configs (build-time only): k ∈ {16, 32, 64}, max_iterations ∈ {10, 20} → 6 builds per library
- Query sweep: max_candidates ∈ {32, 64, 128} → 3 query points
- Reps: 3
- Two libraries × 6 builds × 3 query points = 36 measurements
- **Why glove-25**: NND.jl took 374s on glove-25 vs 691s on glove-100. Tractable. Also a stress test of NND on a 1M-point dataset.
- **Estimated runtime**: NND.jl ~6 × 400s = 40 min, MANN-NND ~6 × 100s = 10 min. **~1 h total.**

#### MANN-KDTree vs SciPy-KDTree (no native Julia KD-tree comparison since SciPy is the standard)
- Already in Tier 1 — drop from Tier 2.

#### Optional: MANN-RPForest vs (Annoy is the closest analogue, already in Tier 1)
- Native Julia RPForest packages exist but are dormant/unmaintained — skip.

### Tier 2 total runtime
- **~2-3 hours** wall-clock if both run on one VM.

---

## Total estimated work

- Tier 1: ~12-18 hours sequentially
- Tier 2: ~2-3 hours sequentially
- **Combined: ~14-21 hours** if one VM did everything

Caveat: gist dominates Tier 1. If gist turns out to be 6h+ alone, that's the long pole — needs to land on a single VM and not block other shards.

## Open prerequisites before running

1. **Refactor merge** — the `query_sweep` refactor must be validated locally before any cloud run.
2. **MANN-RPForest wrapper** — to be written. ~1 hour.
3. **Decide drop list** — confirm dropping HNSW-jl/NND-jl from Tier 1 datasets in YAML configs.
4. **Update existing YAML configs** — add `build:`/`query_sweep:` blocks for sweepable methods.
5. **Capture refactor baseline** — run on fashion-mnist with the new harness before any cloud work, to confirm validation passes.

## Shard allocation

**Target: 7-10 VMs for ~2-3h wall-clock.**

### Shard granularity
Shards = `(dataset, method-group)` where method-group is one of:
- `mann-hnsw-sweep`
- `hnswlib-sweep`
- `mann-lsh-sweep`
- `faiss-ivf-sweep`
- `annoy-sweep`
- `mann-rpforest-sweep`
- `single-point-methods` (BruteForce, KDTree, IVF-HNSW, NND-deferred, PyNNDescent — one shard since each is just one build)

That's up to 7 shards × 9 datasets = **63 Tier 1 shards** + 2 Tier 2 shards = **~65 shards total**.

### Allocation principle
**Build time is the unit of cost.** Distribute slow shards across VMs first, fill with fast ones. Specifically:
1. Sort shards by estimated build time (descending)
2. Assign each in turn to the VM with lowest current load
3. The longest-running VM determines total wall-clock

### Slow shards to spread carefully
- gist + mann-hnsw-sweep (~3-4 h alone)
- gist + hnswlib-sweep (~1-2 h)
- gist + faiss-ivf-sweep (~1 h)
- glove-100 + mann-hnsw-sweep (~1.5-2 h)
- glove-100 + hnswlib-sweep (~1 h)
- nytimes + tier-2-hnsw-headtohead (~1.5 h, HNSW.jl single-threaded)
- glove-25 + tier-2-nnd-headtohead (~1 h, NND.jl is slow)

These 7 slow shards should each go to a different VM. With 7-10 VMs that works directly.

### Fast shards (under 30 min each) — fill the gaps
Most (small_dataset, method-group) combos are fast: fashion-mnist, mnist, lastfm, sift, and most non-HNSW methods on glove/nytimes. They pack onto whichever VM has spare time.

### Concrete allocation (example for 8 VMs)
To be drawn up once the user confirms N. The above slow shards drive the wall-clock; fast ones tile in.
