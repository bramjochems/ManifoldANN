# Manual VM allocation for full thesis run

10 VMs (D16s_v6 in westeurope), each runs its shards sequentially in
fast-first order so the slow ones come last (fail-fast principle).

Total work ~880 min. Target ~88 min/VM. Floor is the longest single
shard (gist-mann-hnsw-sweep, ~100 min).

Estimates per shard from `scripts/thesis/shard_runtime_estimates.md`.

## VM 1 — gist HNSW heavy (~110 min)
1. gist-mann-lsh-sweep (2 min)
2. gist-faiss-ivf-sweep (1 min)
3. gist-mann-rpforest-sweep (1 min)
4. gist-mann-ivf-flat-sweep (8 min)
5. gist-mann-hnsw-sweep (100 min) ★ slowest

## VM 2 — gist + glove-100 HNSW (~100 min)
1. fashion-mnist-mann-lsh-sweep (1)
2. fashion-mnist-faiss-ivf-sweep (1)
3. fashion-mnist-annoy-sweep (1)
4. gist-annoy-sweep (3)
5. gist-hnswlib-sweep (50) ★
6. glove-100-mann-hnsw-sweep (29) ★

## VM 3 — glove-25 NND variants (~100 min)
Most NND-heavy VM. Glove-25 nnd shards are slow due to MANN-NND-full-deferred.
1. fashion-mnist-mann-hnsw-sweep (3)
2. fashion-mnist-hnswlib-sweep (2)
3. mnist-mann-lsh-sweep (1)
4. glove-25-tier2-nnd-mann (25)
5. glove-25-mann-nnd-sweep (25)
6. glove-25-tier2-nnd-jl (40) ★ NND.jl

## VM 4 — gist single-point + IVF-HNSW (~85 min)
1. lastfm-mann-lsh-sweep (1)
2. lastfm-faiss-ivf-sweep (1)
3. lastfm-annoy-sweep (1)
4. lastfm-mann-rpforest-sweep (1)
5. lastfm-mann-hnsw-sweep (3)
6. mnist-mann-rpforest-sweep (1)
7. gist-single-point-methods (30)
8. gist-mann-ivf-hnsw-sweep (30) ★

## VM 5 — nytimes HNSW.jl (~85 min)
1. fashion-mnist-mann-rpforest-sweep (1)
2. mnist-mann-hnsw-sweep (3)
3. mnist-hnswlib-sweep (2)
4. mnist-faiss-ivf-sweep (1)
5. nytimes-mann-lsh-sweep (1)
6. nytimes-faiss-ivf-sweep (1)
7. nytimes-annoy-sweep (1)
8. nytimes-mann-rpforest-sweep (1)
9. nytimes-tier2-hnsw-mann (9)
10. nytimes-tier2-hnsw-jl (55) ★ HNSW.jl

## VM 6 — glove-100 NND + IVF-HNSW (~85 min)
1. mnist-annoy-sweep (1)
2. lastfm-hnswlib-sweep (2)
3. lastfm-mann-nnd-sweep (7)
4. nytimes-mann-nnd-sweep (7)
5. glove-100-mann-ivf-hnsw-sweep (30) ★
6. glove-100-mann-nnd-sweep (32) ★

## VM 7 — sift heavy (~80 min)
1. fashion-mnist-single-point-methods (3)
2. mnist-single-point-methods (3)
3. lastfm-single-point-methods (5)
4. sift-mann-lsh-sweep (1)
5. sift-faiss-ivf-sweep (1)
6. sift-annoy-sweep (1)
7. sift-mann-rpforest-sweep (1)
8. sift-single-point-methods (8)
9. sift-mann-ivf-flat-sweep (5)
10. sift-mann-hnsw-sweep (17)
11. sift-mann-ivf-hnsw-sweep (25) ★
12. sift-mann-nnd-sweep (10)

## VM 8 — glove series (~80 min)
1. nytimes-mann-hnsw-sweep (10)
2. nytimes-hnswlib-sweep (5)
3. nytimes-single-point-methods (4)
4. glove-25-mann-lsh-sweep (1)
5. glove-25-faiss-ivf-sweep (1)
6. glove-25-annoy-sweep (1)
7. glove-25-mann-rpforest-sweep (1)
8. glove-25-mann-hnsw-sweep (10)
9. glove-25-hnswlib-sweep (5)
10. glove-25-single-point-methods (6)
11. glove-50-single-point-methods (7)
12. glove-50-mann-nnd-sweep (27) ★

## VM 9 — glove-50/100 mid (~75 min)
1. glove-50-mann-lsh-sweep (1)
2. glove-50-faiss-ivf-sweep (1)
3. glove-50-annoy-sweep (1)
4. glove-50-mann-rpforest-sweep (1)
5. glove-50-mann-hnsw-sweep (14)
6. glove-50-hnswlib-sweep (7)
7. glove-100-mann-lsh-sweep (1)
8. glove-100-faiss-ivf-sweep (1)
9. glove-100-annoy-sweep (2)
10. glove-100-mann-rpforest-sweep (1)
11. glove-100-mann-ivf-flat-sweep (5)
12. glove-100-hnswlib-sweep (15)
13. glove-100-single-point-methods (10)

## VM 10 — sift HNSW + remaining (~30 min)
1. sift-hnswlib-sweep (9)
2. (any spillover here)

## Allocated count check
VM totals: 5+6+6+8+10+6+12+12+13+1 = 79 shards ✓

## Wall-clock estimate
Floor: longest VM ~110 min (~1h50m).
With 10 VMs in parallel: total wall-clock ~110 min.
Plus VM startup ~5 min once per VM.
Real expected wall-clock: ~2h.

## Cost
~$0.65/hr × 10 VMs × 2h = ~$13 total compute.
